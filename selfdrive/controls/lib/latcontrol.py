import math
import numpy as np
from selfdrive.controls.lib.pid import PIController
from selfdrive.controls.lib.drive_helpers import MPC_COST_LAT
from selfdrive.controls.lib.lateral_mpc import libmpc_py
from common.numpy_fast import interp
from common.realtime import sec_since_boot
from selfdrive.swaglog import cloudlog
from cereal import car

_DT = 0.01    # 100Hz
_DT_MPC = 0.05  # 20Hz


def calc_states_after_delay(states, v_ego, steer_angle, curvature_factor, steer_ratio, delay):
  states[0].x = v_ego * delay
  states[0].psi = v_ego * curvature_factor * math.radians(steer_angle) / steer_ratio * delay
  return states


def get_steer_max(CP, v_ego):
  return interp(v_ego, CP.steerMaxBP, CP.steerMaxV)


class LatControl(object):
  def __init__(self, CP):
    self.pid = PIController((CP.steerKpBP, CP.steerKpV),
                            (CP.steerKiBP, CP.steerKiV),
                            k_f=CP.steerKf, pos_limit=1.0)
    self.last_cloudlog_t = 0.0
    self.setup_mpc(CP.steerRateCost)

  def setup_mpc(self, steer_rate_cost):
    self.libmpc = libmpc_py.libmpc
    self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, steer_rate_cost)
    
    self.mpc_solution = libmpc_py.ffi.new("log_t *")
    self.cur_state = libmpc_py.ffi.new("state_t *")
    self.mpc_updated = False
    self.mpc_nans = False
    self.cur_state[0].x = 0.0
    self.cur_state[0].y = 0.0
    self.cur_state[0].psi = 0.0
    self.cur_state[0].delta = 0.0

    self.last_mpc_ts = 0.0
    self.angle_steers_des = 0.0
    self.angle_steers_des_mpc = 0.0
    self.angle_steers_des_prev = 0.0
    self.angle_steers_des_time = 0.0

    # For Variable Steering Ratio
    self.lowSteerRatio = 9.0           # Set the lowest steering ratio allowed
    self.vsrWindowLow = 0.3            # Set the tire/car angle low-end used for VSR (vsrWindowLow - is same as lowSteerRatio)
    self.vsrWindowHigh = 0.65          # Set the tire/car angle high-end (vsrWindowHigh + is same as CP.steerRatio / interface.py)
    self.manual_Steering_Offset = 0.0  # Set a steering wheel offset. (Should this be * steering ratio to get the steering wheel angle?)
    self.variableSteerRatio = 0.0      # Used to store the calculated steering ratio
    self.angle_Check = 0.0             # Used for desired tire/car angle
    self.vsrSlope = 0.0                # Used for slope intercept formula
    self.vsrYIntercept = 0.0           # Used for slope intercept formula

  def reset(self):
    self.pid.reset()

  def update(self, active, v_ego, angle_steers, steer_override, d_poly, angle_offset, CP, VM, PL):
    cur_time = sec_since_boot()
    self.mpc_updated = False
    # TODO: this creates issues in replay when rewinding time: mpc won't run
    if self.last_mpc_ts < PL.last_md_ts:
      self.last_mpc_ts = PL.last_md_ts
      self.angle_steers_des_prev = self.angle_steers_des_mpc

      curvature_factor = VM.curvature_factor(v_ego)

      l_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.l_poly))
      r_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.r_poly))
      p_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.p_poly))

      # Prius (and Prime) appears to have a variable steering ratio. Try to account for that
      # Applicable maths:
      #  https://www.desmos.com/calculator
      #  https://www.calculator.net/slope-calculator.html
      #  (steering wheel angle / steering ratio) = tire angle ..
      # So, if the calculation below is determining the tire angle, look for values under about 1.5 degrees
      self.angle_Check = angle_steers - angle_offset
      # Seems like a change to k_f in latcontrol is somewhat permanent so pull it every time
      self.pid.k_f = CP.steerKf
      if abs(self.angle_Check) < self.vsrWindowLow :  # 0.3 degrees, for example
        # Is the remaining wandering a result of static friction somewhere in the steering mechanism?
        #  The smallest force required to start motion, or to overcome static friction, is always greater than the force required to continue the motion.
        # So, let's try breaking it loose at low angles
        # I think small corrections are being issued by OP but aren't forceful enough to break static friction in Prius
        # Eventually, a larger correction is needed which is enough to move the steering wheel
        # Increase kf for this steering correction
        self.pid.k_f *= 7.0       # Kf is currently set to 0.00008 so if that changes, may need to set a number here statically
        self.variableSteerRatio = self.lowSteerRatio                        # Use the lower ratio
      elif self.vsrWindowLow < abs(self.angle_Check) < self.vsrWindowHigh:  # The VSR transition zone
        # Begin the _variable_ part
        # Find the slope of the line from the start of the VSR window to the end of the window - ( m = (y1 - y) / (x1 - x) )
        self.vsrSlope = (self.lowSteerRatio - CP.steerRatio) / (self.vsrWindowLow - self.vsrWindowHigh)
        # Solve for b (y-intercept) - (b = y - mx)
        self.vsrYIntercept = (CP.steerRatio - self.vsrSlope) * self.vsrWindowHigh
        # Use b to find y - (y = mx + b)
        self.variableSteerRatio = (self.vsrSlope * self.angle_Check) + self.vsrYIntercept
        # Increase kf here also to a lesser degree. Could use slope intercept here also maybe
        self.pid.k_f *= 3.0        # May need to play with this value more at 70mph
        if not self.lowSteerRatio <= self.variableSteerRatio <= CP.steerRatio:   # Sanity/safety check
          if self.variableSteerRatio < self.lowSteerRatio:
            self.variableSteerRatio = self.lowSteerRatio    # Reset to the low ratio
          elif self.variableSteerRatio > CP.steerRatio:
            self.variableSteerRatio = CP.steerRatio         # Reset to steerRatio from interface.py
      else:                                                 # The angle is in the quick zone so do nothing
        self.variableSteerRatio = CP.steerRatio             # Use steerRatio from interface.py

      # account for actuation delay
      self.cur_state = calc_states_after_delay(self.cur_state, v_ego, angle_steers, curvature_factor, self.variableSteerRatio, CP.steerActuatorDelay)

      v_ego_mpc = max(v_ego, 5.0)  # avoid mpc roughness due to low speed
      self.libmpc.run_mpc(self.cur_state, self.mpc_solution,
                          l_poly, r_poly, p_poly,
                          PL.PP.l_prob, PL.PP.r_prob, PL.PP.p_prob, curvature_factor, v_ego_mpc, PL.PP.lane_width)

      # reset to current steer angle if not active or overriding
      if active:
        delta_desired = self.mpc_solution[0].delta[1]
      else:                   # Add a steering offset vs recalibrating steering sensor so it reads near 0
        delta_desired = math.radians(self.angle_Check) / self.variableSteerRatio
      self.cur_state[0].delta = delta_desired

      self.angle_steers_des_mpc = float(math.degrees(delta_desired * self.variableSteerRatio) + angle_offset + self.manual_Steering_Offset)
      self.angle_steers_des_time = cur_time
      self.mpc_updated = True

      #  Check for infeasable MPC solution
      self.mpc_nans = np.any(np.isnan(list(self.mpc_solution[0].delta)))
      t = sec_since_boot()
      if self.mpc_nans:
        self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, CP.steerRateCost)
        self.cur_state[0].delta = math.radians(angle_steers) / self.variableSteerRatio

        if t > self.last_cloudlog_t + 5.0:
          self.last_cloudlog_t = t
          cloudlog.warning("Lateral mpc - nan: True")

    if v_ego < 0.3 or not active:
      output_steer = 0.0
      self.pid.reset()
    else:
      # TODO: ideally we should interp, but for tuning reasons we keep the mpc solution
      # constant for 0.05s.
      #dt = min(cur_time - self.angle_steers_des_time, _DT_MPC + _DT) + _DT  # no greater than dt mpc + dt, to prevent too high extraps
      #self.angle_steers_des = self.angle_steers_des_prev + (dt / _DT_MPC) * (self.angle_steers_des_mpc - self.angle_steers_des_prev)
      self.angle_steers_des = self.angle_steers_des_mpc
      steers_max = get_steer_max(CP, v_ego)
      self.pid.pos_limit = steers_max
      self.pid.neg_limit = -steers_max
      steer_feedforward = self.angle_steers_des   # feedforward desired angle
      if CP.steerControlType == car.CarParams.SteerControlType.torque:
        steer_feedforward *= v_ego**2  # proportional to realigning tire momentum (~ lateral accel)
      deadzone = 0.0
      output_steer = self.pid.update(self.angle_steers_des, angle_steers, check_saturation=(v_ego > 10), override=steer_override,
                                     feedforward=steer_feedforward, speed=v_ego, deadzone=deadzone)

    self.sat_flag = self.pid.saturated
    return output_steer, float(self.angle_steers_des)
