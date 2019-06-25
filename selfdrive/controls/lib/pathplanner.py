import zmq
import math
import json
import numpy as np
from selfdrive.kegman_conf import kegman_conf
from common.realtime import sec_since_boot
from selfdrive.services import service_list
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.lateral_mpc import libmpc_py
from selfdrive.controls.lib.drive_helpers import MPC_COST_LAT
from selfdrive.controls.lib.model_parser import ModelParser
import selfdrive.messaging as messaging

_DT_MPC = 0.05

def calc_states_after_delay(states, v_ego, steer_angle, curvature_factor, steer_ratio, delay, long_offset):
  states[0].x = v_ego * delay + long_offset
  states[0].delta = math.radians(steer_angle) / steer_ratio
  states[0].psi = curvature_factor * states[0].x * states[0].delta

  return states

def apply_deadzone(angle_steers, angle_steers_des, deadzone):
  if abs(angle_steers_des - angle_steers) <= deadzone:
    return angle_steers_des
  elif angle_steers > angle_steers_des:
    return angle_steers - deadzone
  else:
    return angle_steers + deadzone

class PathPlanner(object):
  def __init__(self, CP):
    self.MP = ModelParser()
    kegman = kegman_conf(CP)
    self.frame = 0
    self.lane_cost = MPC_COST_LAT.LANE
    self.heading_cost = MPC_COST_LAT.HEADING
    self.path_cost = MPC_COST_LAT.PATH
    self.rate_cost = CP.steerRateCost

    self.last_cloudlog_t = 0

    context = zmq.Context()
    self.plan = messaging.pub_sock(context, service_list['pathPlan'].port)
    self.livempc = messaging.pub_sock(context, service_list['liveMpc'].port)
    self.liveStreamData = messaging.pub_sock(context, 8600)

    self.setup_mpc(CP.steerRateCost, self.path_cost, self.lane_cost, self.heading_cost)
    self.invalid_counter = 0
    self.mpc_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    self.mpc_rates = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    self.mpc_times = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    self.mpc_probs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

  def live_tune(self, CP):
    self.frame += 1
    if self.frame % 60 == 0:
      # live tuning through /data/openpilot/tune.py overrides interface.py settings
      kegman = kegman_conf()
      lane_cost = float(kegman.conf['laneCost'])
      path_cost = float(kegman.conf['pathCost'])
      rate_cost = float(kegman.conf['rateCost'])
      heading_cost = float(kegman.conf['headCost'])
      if lane_cost != self.lane_cost or path_cost != self.path_cost or heading_cost != self.heading_cost or rate_cost != self.rate_cost:
        self.lane_cost = lane_cost
        self.path_cost = path_cost
        self.heading_cost = heading_cost
        self.rate_cost = rate_cost
        self.setup_mpc(rate_cost, path_cost, lane_cost, heading_cost)


  def setup_mpc(self, steer_rate_cost, path_cost, lane_cost, heading_cost):
    self.libmpc = libmpc_py.libmpc
    self.libmpc.init(path_cost, lane_cost, heading_cost, steer_rate_cost)
    print(steer_rate_cost, path_cost, lane_cost, heading_cost)

    self.mpc_solution = libmpc_py.ffi.new("log_t *")
    self.cur_state = libmpc_py.ffi.new("state_t *")
    self.cur_state[0].x = 0.0
    self.cur_state[0].y = 0.0
    self.cur_state[0].psi = 0.0
    self.cur_state[0].delta = 0.0

    self.angle_steers_des = 0.0
    self.angle_steers_des_mpc = 0.0

  def update(self, CP, VM, CS, md, live100, live_parameters, live_map_data):
    self.live_tune(CP)
    v_ego = CS.carState.vEgo
    angle_steers = live100.live100.dampAngleSteers
    v_curv = live100.live100.curvature
    delaySteer = live100.live100.delaySteer
    longOffset = live100.live100.longOffset

    angle_offset_average = live_parameters.liveParameters.angleOffsetAverage
    angle_offset_bias = live100.live100.angleModelBias + angle_offset_average

    self.MP.update(v_ego, md, live_map_data, v_curv)

    # Run MPC
    self.angle_steers_des_prev = live100.live100.dampAngleSteersDes
    VM.update_params(live_parameters.liveParameters.stiffnessFactor, live_parameters.liveParameters.steerRatio)

    curvature_factor = VM.curvature_factor(v_ego)

    l_poly = libmpc_py.ffi.new("double[4]", list(self.MP.l_poly))
    r_poly = libmpc_py.ffi.new("double[4]", list(self.MP.r_poly))
    p_poly = libmpc_py.ffi.new("double[4]", list(self.MP.p_poly))

    # account for actuation delay
    self.cur_state = calc_states_after_delay(self.cur_state, v_ego, angle_steers - angle_offset_average, curvature_factor, VM.sR, delaySteer, longOffset)

    v_ego_mpc = max(v_ego, 5.0)  # avoid mpc roughness due to low speed
    self.libmpc.run_mpc(self.cur_state, self.mpc_solution,
                        l_poly, r_poly, p_poly,
                        self.MP.l_prob, self.MP.r_prob, self.MP.p_prob, curvature_factor, v_ego_mpc, self.MP.lane_width)

    # reset to current steer angle if not active or overriding
    #if active:
    self.mpc_angles[0] = angle_steers + live100.live100.angleModelBias
    self.mpc_times[0] = live100.logMonoTime * 1e-9
    oversample_limit = 19 if v_ego == 0 else 4 + min(15, int(800.0 / v_ego))
    for i in range(1,20):
      if i < 6:
        self.mpc_times[i] = self.mpc_times[i-1] + _DT_MPC
        self.mpc_rates[i-1] = (float(math.degrees(self.mpc_solution[0].rate[i-1] * VM.sR)) * self.MP.c_prob \
                               + self.mpc_rates[i] * self.mpc_probs[i]) / (self.MP.c_prob + self.mpc_probs[i])
        self.mpc_probs[i-1] = (self.MP.c_prob**2 + self.mpc_probs[i]**2) / (self.MP.c_prob + self.mpc_probs[i])
      elif i <= oversample_limit:
        self.mpc_times[i] = self.mpc_times[i-1] + 3.0 * _DT_MPC
        self.mpc_rates[i-1] = (float(math.degrees(self.mpc_solution[0].rate[i-1] * VM.sR)) * self.MP.c_prob \
                    + 0.33 * self.mpc_rates[i] * self.mpc_probs[i] \
                    + 0.66 * self.mpc_rates[i-1] * self.mpc_probs[i-1]) \
                    / (self.MP.c_prob + 0.66 * self.mpc_probs[i-1] + 0.33 * self.mpc_probs[i])
        self.mpc_probs[i-1] = (self.MP.c_prob**2 + 0.33 * self.mpc_probs[i]**2 + 0.66 * self.mpc_probs[i-1]**2) \
                    / (self.MP.c_prob + 0.66 * self.mpc_probs[i-1] + 0.33 * self.mpc_probs[i])
      else:
        self.mpc_times[i] = self.mpc_times[i-1] + 3.0 * _DT_MPC
        self.mpc_rates[i-1] = float(math.degrees(self.mpc_solution[0].rate[i-1] * VM.sR))
        self.mpc_probs[i-1] = self.MP.c_prob
      self.mpc_angles[i] = (self.mpc_times[i] - self.mpc_times[i-1]) * self.mpc_rates[i-1] + self.mpc_angles[i-1]

    rate_desired = math.degrees(self.mpc_solution[0].rate[0] * VM.sR)

    self.angle_steers_des_mpc = self.mpc_angles[1]

    #  Check for infeasable MPC solution
    mpc_nans = np.any(np.isnan(list(self.mpc_solution[0].delta)))
    t = sec_since_boot()
    if mpc_nans:
      self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, CP.steerRateCost)
      self.cur_state[0].delta = math.radians(angle_steers) / VM.sR

      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Lateral mpc - nan: True")

    if self.mpc_solution[0].cost > 20000. or mpc_nans:   # TODO: find a better way to detect when MPC did not converge
      self.invalid_counter += 1
    else:
      self.invalid_counter = 0

    plan_valid = self.invalid_counter < 2

    plan_send = messaging.new_message()
    plan_send.init('pathPlan')
    plan_send.pathPlan.laneWidth = float(self.MP.lane_width)
    plan_send.pathPlan.dPoly = map(float, self.MP.d_poly)
    plan_send.pathPlan.cPoly = map(float, self.MP.c_poly)
    plan_send.pathPlan.cProb = float(self.MP.c_prob)
    plan_send.pathPlan.lPoly = map(float, l_poly)
    plan_send.pathPlan.lProb = float(self.MP.l_prob)
    plan_send.pathPlan.rPoly = map(float, r_poly)
    plan_send.pathPlan.rProb = float(self.MP.r_prob)
    plan_send.pathPlan.angleSteers = float(self.angle_steers_des_mpc)
    plan_send.pathPlan.rateSteers = float(rate_desired)
    plan_send.pathPlan.mpcAngles = map(float, self.mpc_angles)
    plan_send.pathPlan.mpcRates = map(float, self.mpc_rates)
    plan_send.pathPlan.mpcTimes = map(float, self.mpc_times)
    plan_send.pathPlan.laneProb =float(self.MP.lane_prob)
    plan_send.pathPlan.angleOffset = float(angle_offset_average)
    plan_send.pathPlan.valid = bool(plan_valid)
    plan_send.pathPlan.paramsValid = bool(live_parameters.liveParameters.valid)

    self.plan.send(plan_send.to_bytes())

    dat = messaging.new_message()
    dat.init('liveMpc')
    dat.liveMpc.x = list(self.mpc_solution[0].x)
    dat.liveMpc.y = list(self.mpc_solution[0].y)
    dat.liveMpc.psi = list(self.mpc_solution[0].psi)
    dat.liveMpc.delta = list(self.mpc_solution[0].delta)
    dat.liveMpc.cost = self.mpc_solution[0].cost
    self.livempc.send(dat.to_bytes())

    if len(live_map_data.liveMapData.roadCurvature) > 0:
      curv_data = "%d,%d,%d,%d,%d,%d,%d,%d,%d," % (self.MP.l_curv, self.MP.p_curv, self.MP.r_curv, self.MP.map_curv, self.MP.map_rcurv, self.MP.map_rcurvx, self.MP.v_curv2, self.MP.l_diverge, self.MP.r_diverge)
      #print(live_map_data.liveMapData.roadCurvature)
      self.liveStreamData.send_string(curv_data)
