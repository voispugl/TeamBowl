#!/usr/bin/env python3
"""
TeamBowl MuJoCo Balance Simulation
====================================
Standalone two-wheeled self-balancing robot sim using cascaded PID.

Architecture
------------
  Physics   : 500 Hz  (timestep = 0.002 s)
  Inner PD  : 150 Hz  — complementary-filter pitch → torque
  Outer PI  : 40  Hz  — velocity error → target pitch angle

Sensors used
------------
  imu_gyro    : body-frame angular velocity  [wx, wy, wz]
  imu_accel   : body-frame linear acceleration [ax, ay, az]
  gt_pos      : world-frame position [x, y, z]           (ground truth)
  gt_quat     : world-frame orientation [w, x, y, z]     (ground truth)
  gt_linvel   : world-frame linear velocity [vx, vy, vz] (ground truth)
  gt_angvel   : world-frame angular velocity              (ground truth)

The gt_* sensors are the "perfect EKF" — they validate the complementary
filter estimate in real time (see pitch_cf vs pitch_gt in the CSV).

Axes / sign convention
-----------------------
  Wheel axle  : world X   (wheels spin around X)
  Forward     : world Y   (positive Y = forward)
  Pitch axis  : Y-axis (gyro[1]) — matches real IMU mounting convention.
  Positive pitch  = leaning forward (nose down).

  NOTE: If the robot falls backward, flip PITCH_SIGN = -1.

Live terminal gain tuning
--------------------------
Type these commands while the sim is running (hit Enter):
  kp=80       set KP_PITCH
  kd=10       set KD_PITCH
  kpv=0.4     set KP_VEL
  kiv=0.06    set KI_VEL
  v=0.5       set target forward velocity V_CMD (m/s)
  gains       print current gain values
  reset       zero V_CMD and reset velocity integrator
  autotune    start Bayesian Optimization (30 trials)

Output
------
  teambowl_sim_log.csv  — logged every LOG_EVERY inner ticks
  MuJoCo passive viewer window

Usage
-----
  pip install mujoco scikit-optimize
  python sim.py
"""

import math
import sys
import threading
import time
import csv
import os
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# =============================================================================
# TUNING MANAGER — Handles Bayesian Optimization trials
# =============================================================================

class TuningManager:
    def __init__(self, duration=8.0):
        self.active = False
        self.duration = duration
        self.trial_start_time = 0.0
        self.finished_event = threading.Event()
        self.trial_number = 0

        # Performance metrics
        self.pitch_error_abs_sum = 0.0
        self.vel_error_abs_sum = 0.0
        self.torque_abs_sum = 0.0
        self.fell = False
        self.step_count = 0
        self.foot_contact_steps = 0

        self.last_cost = 0.0

    def reset_trial(self, t_sim):
        self.trial_number += 1
        self.trial_start_time = t_sim
        self.pitch_error_abs_sum = 0.0
        self.vel_error_abs_sum = 0.0
        self.torque_abs_sum = 0.0
        self.fell = False
        self.step_count = 0
        self.foot_contact_steps = 0
        # NOTE: finished_event is cleared in objective() before active=True,
        # not here, so the main loop guard (not finished_event.is_set()) works.

    def update(self, t_sim, pitch_err, vel_err, torque, fell, foot_contact):
        if not self.active:
            return

        self.pitch_error_abs_sum += abs(pitch_err)
        self.vel_error_abs_sum += abs(vel_err)
        self.torque_abs_sum += abs(torque)
        self.step_count += 1

        if fell:
            self.fell = True
        if foot_contact:
            self.foot_contact_steps += 1

        if (t_sim - self.trial_start_time) >= self.duration or self.fell:
            self.calculate_cost(t_sim)
            self.finished_event.set()

    def calculate_cost(self, t_sim):
        if self.step_count == 0:
            self.last_cost = 3000.0
            return

        # Normalized metrics
        avg_pitch_err = self.pitch_error_abs_sum / self.step_count
        avg_vel_err   = self.vel_error_abs_sum   / self.step_count
        avg_torque    = self.torque_abs_sum       / self.step_count

        # Penalty for falling or ending early
        survival_time = t_sim - self.trial_start_time
        time_penalty  = (self.duration - survival_time) * 200.0
        fall_penalty  = 1000.0 if self.fell else 0.0

        # Bonus for surviving the full trial without falling
        no_fall_bonus = -500.0 if not self.fell else 0.0

        # Total cost (lower is better)
        self.last_cost = (
            avg_pitch_err  * 150.0 +
            avg_vel_err    *  50.0 +
            avg_torque     *   0.05 +
            time_penalty +
            fall_penalty +
            no_fall_bonus
        )

_tuning = TuningManager()

# =============================================================================
# OPTIMIZER — Bayesian search space and objective function
# =============================================================================

# Search space: KP_PITCH, KD_PITCH, KI_PITCH, KP_VEL, KI_VEL
# Narrowed bounds for "smaller steps" around starting values
SPACE = [
    # Widened bounds to cover the full plausible hardware range
    Real(50.0,  2000.0, name='kp_pitch'),  # sim working value ~800-1600
    Real(0.0,   500.0,  name='kd_pitch'),
    Real(0.0,   600.0,  name='ki_pitch'),
    Real(0.0,   30.0,   name='kp_vel'),
    Real(0.0,   20.0,   name='ki_vel'),
    Real(0.0,   1.0,    name='kff_pitch'), # feed-forward lean  [rad/(m/s)]
]

# Number of independent restarts — each uses different random seeds so the
# GP explores different regions; we keep the best result across all runs.
N_RESTARTS   = 3
CALLS_PER_RUN = 150   # trials per restart  (3 × 150 = 450 total)
TRIAL_DURATION = 15.0  # seconds per trial

@use_named_args(SPACE)
def objective(**params):
    # Set gains for this trial
    for name, value in params.items():
        _set_gain(name, value)
    _set_gain("v_cmd", 0.2)

    # Clear event BEFORE setting active so the main loop guard works correctly:
    # main loop checks (active AND NOT finished_event) to start a trial.
    _tuning.finished_event.clear()
    _tuning.active = True
    _tuning.finished_event.wait()   # blocks until main loop signals trial done
    _tuning.active = False

    cost = _tuning.last_cost
    n    = _tuning.trial_number
    print(f"[autotune] Trial {n:3d} | cost={cost:8.2f} | "
          f"kp={params['kp_pitch']:6.1f} ki={params['ki_pitch']:6.1f} kd={params['kd_pitch']:6.1f} | "
          f"kpv={params['kp_vel']:5.2f} kiv={params['ki_vel']:5.2f} kff={params['kff_pitch']:5.3f}")
    return cost

def _optimizer_thread():
    print("\n" + "="*75)
    print(f"  AUTOTUNE: {N_RESTARTS} independent runs × {CALLS_PER_RUN} trials")
    print(f"  Trial duration: {TRIAL_DURATION} s  |  100 random init points per run")
    print("="*75 + "\n")

    _tuning.duration = TRIAL_DURATION

    # Starting point from current constants
    x0 = [KP_PITCH, KD_PITCH, KI_PITCH, KP_VEL, KI_VEL, KFF_PITCH]

    best_res  = None
    best_cost = float("inf")

    for run in range(N_RESTARTS):
        import numpy as np
        seed = 42 + run * 17   # different seed each restart for diverse exploration
        print(f"\n[autotune] === Run {run+1}/{N_RESTARTS} (seed={seed}) ===")

        res = gp_minimize(
            objective,
            SPACE,
            n_calls=CALLS_PER_RUN,
            n_initial_points=100,  # heavy random exploration before GP fits
            x0=x0 if run == 0 else None,  # seed from current gains on first run only
            noise=0.01,            # higher noise → less confident GP → more exploration
            random_state=seed,
            verbose=False,
        )

        print(f"[autotune] Run {run+1} best cost: {res.fun:.2f}  gains: {[f'{v:.2f}' for v in res.x]}")

        if res.fun < best_cost:
            best_cost = res.fun
            best_res  = res

    print("\n" + "="*75)
    print("  OPTIMIZATION FINISHED!")
    print(f"  Best cost across all runs: {best_cost:.2f}")
    names = [s.name for s in SPACE]
    for name, val in zip(names, best_res.x):
        print(f"    {name} = {val:.4f}")
    print("="*75 + "\n")

    # Apply best gains found
    for name, val in zip(names, best_res.x):
        _set_gain(name, val)
    _set_gain("v_cmd", 0.0)
    _set_gain("reset", True)
    _tuning.duration = TRIAL_DURATION  # restore default duration for manual use

import mujoco
import mujoco.viewer
from collections import deque

# =============================================================================
# GAINS — edit here or update live from the terminal
# =============================================================================

# Inner PID — pitch error → wheel torque (Nm)
KP_PITCH   = 80.0    # proportional gain  [Nm / rad]
KD_PITCH   = 100.0     # derivative gain    [Nm·s / rad]
KI_PITCH      = 200.0     # integral gain  [Nm / (rad·s)] — start at 0
PITCH_I_WINDOW = 1.0    # seconds of pitch integral memory

# Outer PI — velocity error → target pitch (rad)
KP_VEL     = 10    # proportional gain  [rad / (m/s)]
KI_VEL     = 5    # integral gain      [rad / (m·s)]
KFF_PITCH  = 0.0   # feed-forward: lean setpoint ∝ v_cmd  [rad / (m/s)]

# Safety / saturation limits
THETA_MAX     = 1    # max commanded lean angle  [rad]
TORQUE_MAX    = 40.0    # per-wheel torque cap      [Nm]
THETA_FALLOVER = 0.90   # estop threshold           [rad]
I_WIND_MAX    = THETA_MAX / max(KI_VEL, 1e-6)  # anti-windup integrator clamp

# Complementary filter — weight on gyro vs accelerometer
ALPHA = 0.98

# Yaw rate PD — differential torque between wheels
KP_YAW    = 0    # yaw rate proportional  [Nm / (rad/s)]
KD_YAW    = 0    # yaw rate derivative     [Nm·s² / rad]

# Target forward velocity (m/s) — change with 'v=X' in terminal
V_CMD = 0.0

# Commanded yaw rate (rad/s) — change with 'omega=X' in terminal
OMEGA_CMD = 0.0

# Pitch sign — flipped to -1 because the robot was falling the opposite direction
PITCH_SIGN = -1.0

# Yaw sign — flip to -1 if turning is backwards
YAW_SIGN = -1.0

# Control loop frequencies (derived from physics dt = 0.002 s)
DT_PHYSICS   = 0.002                      # physics timestep [s]
DT_INNER     = 1.0 / 150.0               # ~0.00667 s → every 4 steps (approx)
DT_OUTER     = 1.0 / 40.0                # 0.025 s   → every 13 steps

# How many inner-loop ticks between CSV log rows
LOG_EVERY = 5

# Initial Z position of the Frame body so wheels rest on the floor.
# Floor at z=-0.3, wheel radius=0.154, wheel body offset from Frame ≈ -0.056
# Frame z = -0.3 + 0.154 + 0.056 = -0.090
INIT_Z = -0.090

# Path to the XML model (relative to this file's directory)
XML_PATH = os.path.join(os.path.dirname(__file__), "teambowl_balance.xml")
CSV_PATH = os.path.join(os.path.dirname(__file__), "teambowl_sim_log.csv")

# =============================================================================
# Shared gain state — protected by a lock so the terminal thread can write
# while the control loop reads.
# =============================================================================

_gains_lock = threading.Lock()
_gains = {
    "kp_pitch":   KP_PITCH,
    "kd_pitch":   KD_PITCH,
    "ki_pitch":   KI_PITCH,
    "kp_vel":     KP_VEL,
    "ki_vel":     KI_VEL,
    "kff_pitch":  KFF_PITCH,
    "kp_yaw":     KP_YAW,
    "kd_yaw":     KD_YAW,
    "v_cmd":      V_CMD,
    "omega_cmd":  OMEGA_CMD,
    "pitch_sign": PITCH_SIGN,  # 1.0 or -1.0 — flip if robot falls wrong way
    "yaw_sign":   YAW_SIGN,    # 1.0 or -1.0 — flip if turning is backwards
    "reset":      False,       # one-shot flag: reset velocity integrator
}


def _get_gains():
    with _gains_lock:
        return dict(_gains)


def _set_gain(key, value):
    with _gains_lock:
        _gains[key] = value


# =============================================================================
# Terminal gain-tuning thread
# =============================================================================

def _terminal_thread():
    """
    Reads lines from stdin and updates _gains.
    Runs as a daemon so it exits when the main thread ends.

    Commands:
      kp= kd= ki= kpv= kiv= v= omega= | gains reset autotune
    """
    ALIASES = {
        "kp":    "kp_pitch",
        "kd":    "kd_pitch",
        "ki":    "ki_pitch",
        "kpv":   "kp_vel",
        "kiv":   "ki_vel",
        "kff":   "kff_pitch",
        "kyp":   "kp_yaw",
        "kyd":   "kd_yaw",
        "v":     "v_cmd",
        "omega": "omega_cmd",
    }
    TUNABLE = {"kp_pitch", "kd_pitch", "ki_pitch", "kp_vel", "ki_vel", "kff_pitch",
               "kp_yaw", "kd_yaw", "v_cmd", "omega_cmd"}
    print("[gains] kp= kd= ki= kpv= kiv= v= omega= | gains reset autotune")
    for line in sys.stdin:
        line = line.strip().lower()
        if not line:
            continue
        if line == "autotune":
            threading.Thread(target=_optimizer_thread, daemon=True).start()
            continue
        if line == "gains":
            g = _get_gains()
            print(
                f"[gains] kp={g['kp_pitch']:.1f}  ki={g['ki_pitch']:.3f}  kd={g['kd_pitch']:.1f}  "
                f"kpv={g['kp_vel']:.3f}  kiv={g['ki_vel']:.3f}  "
                f"kyp={g['kp_yaw']:.3f}  kyd={g['kd_yaw']:.3f}  "
                f"v={g['v_cmd']:.3f}  ω={g['omega_cmd']:.3f}"
            )
            continue
        if line == "reset":
            with _gains_lock:
                _gains["v_cmd"]     = 0.0
                _gains["omega_cmd"] = 0.0
                _gains["reset"]     = True
            print("[gains] Reset: v=0, ω=0, integrator cleared")
            continue
        if "=" in line:
            key_raw, _, val_str = line.partition("=")
            key_raw  = key_raw.strip()
            full_key = ALIASES.get(key_raw, key_raw)
            try:
                val = float(val_str.strip())
                if full_key in TUNABLE:
                    _set_gain(full_key, val)
                    print(f"[gains] {full_key} = {val:.4f}")
                else:
                    print(f"[gains] Unknown: '{full_key}'")
            except ValueError:
                print(f"[gains] Bad value: '{val_str}'")


# =============================================================================
# Helper: extract pitch angle from world-frame quaternion
# =============================================================================

def _quat_to_pitch_gt(qw, qx, qy, qz):
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    return math.asin(sinp)


def _quat_to_rpy(qw, qx, qy, qz):
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


# =============================================================================
# Main simulation
# =============================================================================

def main():
    print(f"[sim] Loading model: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    data.qpos[2] = INIT_Z
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)

    def sid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)

    def sensor_data(name):
        s_id = sid(name)
        adr  = model.sensor_adr[s_id]
        dim  = model.sensor_dim[s_id]
        return data.sensordata[adr : adr + dim]

    # Geom IDs for foot-floor contact detection.
    # Foot geoms are all mesh127 geoms with contype=1; floor is the named plane.
    _floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    _foot_geom_ids = set()
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        mesh_id = model.geom_dataid[i]
        if mesh_id >= 0:
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            if mesh_name == "mesh127":
                _foot_geom_ids.add(i)

    def foot_touching_floor():
        for i in range(data.ncon):
            c = data.contact[i]
            if (c.geom1 == _floor_geom_id and c.geom2 in _foot_geom_ids) or \
               (c.geom2 == _floor_geom_id and c.geom1 in _foot_geom_ids):
                return True
        return False

    pitch_cf       = 0.0
    v_integral     = 0.0
    pitch_i_window = deque()

    t_last_inner = 0.0
    t_last_outer = 0.0
    t_last_print = 0.0
    log_tick = 0

    t = threading.Thread(target=_terminal_thread, daemon=True)
    t.start()

    csv_file = open(CSV_PATH, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["t_sim", "pitch_cf_deg", "pitch_gt_deg", "pitch_err_deg", "theta_ref_deg", "torque_Nm", "v_actual_ms", "v_cmd_ms"])

    theta_ref = 0.0
    t_fallen  = None
    trial_in_progress = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_start   = time.time()
        step_count  = 0

        while viewer.is_running():
            t_sim = data.time

            # -- Trial Management --
            # Guard: don't start a new trial until the optimizer has cleared the
            # finished_event (happens in objective() before it sets active=True again).
            # Without this, after a trial ends trial_in_progress=False but active is
            # still True (optimizer thread hasn't woken yet), causing overlapping trials.
            if _tuning.active and not trial_in_progress and not _tuning.finished_event.is_set():
                mujoco.mj_resetData(model, data)
                data.qpos[2] = INIT_Z
                data.qpos[3] = 1.0
                mujoco.mj_forward(model, data)
                pitch_cf = 0.0
                v_integral = 0.0
                pitch_i_window.clear()
                theta_ref = 0.0
                # mj_resetData resets data.time to 0 — must reset loop timers too
                # or t_sim - t_last_* goes negative and loops never fire
                t_last_inner = data.time
                t_last_outer = data.time
                t_last_print = data.time
                sim_start = time.time()
                step_count = 0
                _tuning.reset_trial(data.time)
                trial_in_progress = True
                print(f"[autotune] Trial {_tuning.trial_number:3d} starting...")

            gyro  = sensor_data("imu_gyro")
            accel = sensor_data("imu_accel")
            gt_quat   = sensor_data("gt_quat")
            gt_linvel = sensor_data("gt_linvel")
            gt_pos    = sensor_data("gt_pos")
            gt_angvel = sensor_data("gt_angvel")

            g_sign = _get_gains()["pitch_sign"]
            pitch_gt = g_sign * _quat_to_pitch_gt(gt_quat[0], gt_quat[1], gt_quat[2], gt_quat[3])
            v_actual = gt_linvel[1]
            yaw_rate_actual = float(gt_angvel[2])

            if t_sim - t_last_print >= 1.0 and not _tuning.active:
                t_last_print = t_sim
                roll, pitch, yaw = _quat_to_rpy(gt_quat[0], gt_quat[1], gt_quat[2], gt_quat[3])
                print(f"[t={t_sim:6.1f}s] θ={math.degrees(pitch):+7.2f}° θ_t={math.degrees(theta_ref):+7.2f}° v={v_actual:+.3f}m/s")

            # Outer PI (40 Hz)
            if t_sim - t_last_outer >= DT_OUTER:
                dt_o = t_sim - t_last_outer
                t_last_outer = t_sim
                g = _get_gains()
                if g["reset"]:
                    v_integral = 0.0
                    with _gains_lock: _gains["reset"] = False

                v_err = g["v_cmd"] - v_actual
                v_integral += v_err * dt_o
                ki = g["ki_vel"]
                max_i = THETA_MAX / max(ki, 1e-6)
                v_integral = max(-max_i, min(max_i, v_integral))
                theta_ref = (g["kp_vel"] * v_err
                           + ki * v_integral
                           + g["kff_pitch"] * g["v_cmd"])  # feed-forward lean
                theta_ref = max(-THETA_MAX, min(THETA_MAX, theta_ref))

            # Inner PD (150 Hz)
            torque = 0.0
            if t_sim - t_last_inner >= DT_INNER:
                dt_i = t_sim - t_last_inner
                t_last_inner = t_sim
                g = _get_gains()
                gyro_pitch = g["pitch_sign"] * float(gyro[1])
                ax, ay, az = float(accel[0]), float(accel[1]), float(accel[2])
                accel_pitch = PITCH_SIGN * math.atan2(ax, math.sqrt(ay * ay + az * az))
                pitch_cf = (ALPHA * (pitch_cf + gyro_pitch * dt_i) + (1.0 - ALPHA) * accel_pitch)
                pitch_err = pitch_cf - theta_ref
                pitch_i_window.append((t_sim, pitch_err * dt_i))
                while pitch_i_window and t_sim - pitch_i_window[0][0] > PITCH_I_WINDOW: pitch_i_window.popleft()
                pitch_integral = sum(v for _, v in pitch_i_window)
                torque = -(g["kp_pitch"] * pitch_err + g["ki_pitch"] * pitch_integral + g["kd_pitch"] * gyro_pitch)
                torque = max(-TORQUE_MAX, min(TORQUE_MAX, torque))

                # -- Fallover Reset --
                if abs(pitch_gt) > THETA_FALLOVER:
                    data.ctrl[:] = 0.0
                    if _tuning.active:
                        # Notify tuning manager so the trial can finish (unblocks optimizer thread)
                        _tuning.update(t_sim, pitch_cf - theta_ref, g["v_cmd"] - v_actual, torque, True, foot_touching_floor())
                        if _tuning.finished_event.is_set():
                            trial_in_progress = False
                            mujoco.mj_resetData(model, data)
                            data.qpos[2] = INIT_Z; data.qpos[3] = 1.0
                            mujoco.mj_forward(model, data)
                            pitch_cf = 0.0; v_integral = 0.0; pitch_i_window.clear()
                            sim_start = time.time(); step_count = 0
                    else:
                        if t_fallen is None: t_fallen = t_sim
                        elif t_sim - t_fallen >= 1.0:
                            mujoco.mj_resetData(model, data)
                            data.qpos[2] = INIT_Z; data.qpos[3] = 1.0
                            mujoco.mj_forward(model, data)
                            pitch_cf = 0.0; v_integral = 0.0; pitch_i_window.clear(); t_fallen = None
                            sim_start = time.time(); step_count = 0
                    continue
                t_fallen = None

                yaw_err = g["omega_cmd"] - yaw_rate_actual * g["yaw_sign"]
                yaw_correction = g["kp_yaw"] * yaw_err - g["kd_yaw"] * yaw_rate_actual * g["yaw_sign"]
                data.ctrl[0] = torque - yaw_correction
                data.ctrl[1] = -(torque + yaw_correction)

                # -- Trial Update --
                if trial_in_progress:
                    g = _get_gains()
                    _tuning.update(t_sim, pitch_cf - theta_ref, g["v_cmd"] - v_actual, torque, abs(pitch_gt) > THETA_FALLOVER, foot_touching_floor())
                    if _tuning.finished_event.is_set():
                        trial_in_progress = False
                        mujoco.mj_resetData(model, data)
                        data.qpos[2] = INIT_Z; data.qpos[3] = 1.0; mujoco.mj_forward(model, data)
                        pitch_cf = 0.0; v_integral = 0.0; pitch_i_window.clear()
                        sim_start = time.time(); step_count = 0

                log_tick += 1
                if log_tick >= LOG_EVERY:
                    log_tick = 0
                    writer.writerow([t_sim, math.degrees(pitch_cf), math.degrees(pitch_gt), math.degrees(pitch_cf-pitch_gt), math.degrees(theta_ref), torque, v_actual, g["v_cmd"]])

            mujoco.mj_step(model, data)
            step_count += 1
            # Sync viewer every step normally; during autotune only sync occasionally
            # so physics runs faster than real-time
            if not _tuning.active or step_count % 30 == 0:
                viewer.sync()
            # Run as fast as possible during autotune; throttle to real-time otherwise
            if not _tuning.active:
                elapsed_wall = time.time() - sim_start
                sleep_needed = (step_count * DT_PHYSICS) - elapsed_wall
                if sleep_needed > 0: time.sleep(sleep_needed)

    csv_file.close()
    print(f"\n[sim] Done. Log saved to {CSV_PATH}")

if __name__ == "__main__":
    main()
