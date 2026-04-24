"""
Leg inverse kinematics for the TeamBowl parallel 5-bar legs.

Each leg is a 5-bar mechanism with two actuated joints at the hip:
  Motor A (dof_calf1_0 / calf0_0)   — axis −X, drives the Thigh
  Motor B (dof_driver1_0 / driver0_0) — axis +X, drives Kicker → Align bar

The parallel-bar constraint makes this equivalent to a 2R planar arm
in the sagittal plane:
  Segment 1 (shoulder→knee):  driven by Motor A, length L_THIGH
  Segment 2 (knee→foot):      direction controlled by Motor B via
                               the parallel link, length L_CALF

The hip rotation joint (Y axis) is treated as the 3rd DOF and handled
separately; IK operates in the 2D sagittal plane (Y-Z in Hip frame).

Raw encoder values vs. URDF angles
------------------------------------
The RS04 encoders are absolute but their zero is set at commissioning,
so raw values do NOT align with URDF angle zeros.  This module uses a
per-joint calibration offset:

    θ_urdf = (raw_encoder - encoder_zero) * axis_sign

Call calibrate_from_driving_pos() once at startup to fit the offsets
from the known driving position (raw encoder values from /joint_states)
and a measured or assumed foot height in that configuration.

Units: metres, radians.
"""

import math
import numpy as np
from scipy.optimize import minimize, brentq

# ── Link geometry from URDF ────────────────────────────────────────────────── #

# Hip motor pivot offsets in Hip frame (x, y, z)
_P_A = np.array([-0.03125, -0.07, 0.0])   # Motor A (dof_calf1_0)
_P_B = np.array([+0.03125, -0.07, 0.0])   # Motor B (dof_driver1_0)

# Joint pre-rotations (static offset in URDF origin rpy, about X)
_PRE_A = -0.436332   # dof_calf1_0  rpy[0]
_PRE_B = -0.785398   # dof_driver1_0 rpy[0]

# Pre-rotation of the passive knee joint (closing_knee / knee0)
_PRE_KNEE = -0.349066  # closing_knee1_0 rpy[0]

# Joint axis signs: Motor A has axis=(-1,0,0) → sign=-1; Motor B axis=(+1,0,0) → sign=+1
_SIGN_A = -1.0
_SIGN_B = +1.0

# Offsets in child-link frames (from URDF joint origins)
_KNEE_IN_THIGH  = np.array([ 0.02625, -0.225522,  0.190669])  # closing_knee origin in Thigh
_TROCHO_IN_KICK = np.array([-0.03625, -0.022511,  0.065706])  # trochanter origin in Kicker
_ANKLE_IN_CALF  = np.array([ 0.02725,  0.095716, -0.279381])  # ankle origin in Calf

# Derived link lengths
L_THIGH = float(np.linalg.norm(_KNEE_IN_THIGH))   # ≈ 0.2966 m
L_KICKER = float(np.linalg.norm(_TROCHO_IN_KICK))  # ≈ 0.0784 m
L_CALF   = float(np.linalg.norm(_ANKLE_IN_CALF))   # ≈ 0.2966 m
L_MAX    = L_THIGH + L_CALF                        # ≈ 0.5932 m

# Align-bar effective length (from URDF inertial origin approximation).
# The align bar tip attaches back to the Calf at a point that enforces
# the parallel constraint; length tuned to match URDF inertial origin.
_ALIGN_BAR_REACH = math.sqrt(0.138854**2 + 0.117827**2)  # ≈ 0.182 m


def _rotx(angle: float) -> np.ndarray:
    """3×3 rotation matrix about X axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])


def _transform(origin: np.ndarray, pre_rot: float,
               joint_angle: float, axis_sign: float,
               point: np.ndarray) -> np.ndarray:
    """
    Apply a single revolute joint transform to a point in the child frame,
    returning the point in the parent frame.

    T = Trans(origin) @ Rotx(pre_rot) @ Rotx(axis_sign * joint_angle)
    """
    R = _rotx(pre_rot) @ _rotx(axis_sign * joint_angle)
    return origin + R @ point


# ── Forward Kinematics ─────────────────────────────────────────────────────── #

def leg_fk_urdf(theta_A: float, theta_B: float) -> np.ndarray:
    """
    Compute foot position (x, y, z) in Hip frame given URDF-frame joint angles.

    theta_A: Motor A angle in URDF convention (radians from URDF zero)
    theta_B: Motor B angle in URDF convention (radians from URDF zero)

    Returns foot origin as (3,) array in Hip frame.

    The passive knee angle is solved from the 4-bar closure constraint:
    the trochanter tip (Motor B chain) must lie at the correct position
    relative to the Calf upper pivot to enforce the parallel linkage.
    In a true parallelogram 4-bar, the Calf orientation equals the angle
    of the Kicker arm.  We approximate this by computing the kicker tip
    direction and using it as the calf direction.
    """
    # Motor A chain: Hip → Thigh → knee joint
    P_knee = _transform(_P_A, _PRE_A, theta_A, _SIGN_A, _KNEE_IN_THIGH)

    # Motor B chain: Hip → Kicker → trochanter
    P_trocho = _transform(_P_B, _PRE_B, theta_B, _SIGN_B, _TROCHO_IN_KICK)

    # Parallel constraint: the calf direction (knee→foot unit vector) is
    # parallel to the kicker arm direction.  Kicker arm direction = unit
    # vector from P_B (pivot) to P_trocho.
    kicker_vec = P_trocho - _P_B
    kicker_len = np.linalg.norm(kicker_vec)
    if kicker_len < 1e-9:
        raise ValueError("Kicker degenerate (zero length arm)")
    calf_dir = kicker_vec / kicker_len  # unit vector, same direction as kicker

    # Foot = knee_position + L_CALF * calf_direction
    P_foot = P_knee + L_CALF * calf_dir
    return P_foot


def leg_fk(theta_A_raw: float, theta_B_raw: float,
           zero_A: float, zero_B: float) -> np.ndarray:
    """
    FK from raw encoder values using calibration offsets.

    zero_A, zero_B: raw encoder values when the joint is at URDF angle 0.

    Returns foot position (x, y, z) in Hip frame.
    """
    theta_A = (theta_A_raw - zero_A) * _SIGN_A
    theta_B = (theta_B_raw - zero_B) * _SIGN_B
    return leg_fk_urdf(theta_A, theta_B)


# ── Inverse Kinematics ─────────────────────────────────────────────────────── #

def leg_ik_urdf(target: np.ndarray,
                guess_A: float = 0.0,
                guess_B: float = 0.0,
                tol: float = 1e-4) -> tuple[float, float] | None:
    """
    Solve IK for a desired foot position in Hip frame.

    target:  (3,) foot position in Hip frame
    guess_A/B: initial URDF-frame joint angle guesses
    tol:     convergence tolerance (metres)

    Returns (theta_A, theta_B) in URDF convention, or None if failed.
    """
    def cost(x):
        try:
            p = leg_fk_urdf(x[0], x[1])
        except ValueError:
            return 1e6
        return float(np.sum((p - target) ** 2))

    result = minimize(
        cost,
        x0=[guess_A, guess_B],
        method='L-BFGS-B',
        bounds=[(-2 * math.pi, 2 * math.pi),
                (-2 * math.pi, 2 * math.pi)],
        options={'ftol': tol ** 2, 'gtol': 1e-6, 'maxiter': 500},
    )
    if not result.success and cost(result.x) > tol ** 2 * 10:
        return None
    if math.sqrt(cost(result.x)) > tol * 20:
        return None
    return float(result.x[0]), float(result.x[1])


def leg_ik(target: np.ndarray,
           zero_A: float, zero_B: float,
           guess_A_raw: float | None = None,
           guess_B_raw: float | None = None,
           tol: float = 1e-3) -> tuple[float, float] | None:
    """
    IK in raw encoder space.

    Returns (raw_A, raw_B) or None if no solution found.
    """
    guess_A = (guess_A_raw - zero_A) * _SIGN_A if guess_A_raw is not None else 0.0
    guess_B = (guess_B_raw - zero_B) * _SIGN_B if guess_B_raw is not None else 0.0

    result = leg_ik_urdf(target, guess_A, guess_B, tol)
    if result is None:
        return None
    theta_A_urdf, theta_B_urdf = result
    raw_A = theta_A_urdf / _SIGN_A + zero_A
    raw_B = theta_B_urdf / _SIGN_B + zero_B
    return raw_A, raw_B


# ── Calibration ────────────────────────────────────────────────────────────── #

class LegCalibration:
    """
    Per-leg calibration: maps raw encoder values to URDF joint angles.

    Usage:
        cal = LegCalibration()
        cal.calibrate_from_driving_pos(
            theta_A_raw=0.406, theta_B_raw=2.628,
            foot_height_m=-0.28   # approximate foot Z in Hip frame while driving
        )
        foot_pos = cal.fk(theta_A_raw, theta_B_raw)
        raw_A, raw_B = cal.ik(target_pos, guess_A_raw, guess_B_raw)
    """

    def __init__(self):
        self.zero_A: float = 0.0
        self.zero_B: float = 0.0
        self._calibrated: bool = False

    def calibrate_from_driving_pos(self,
                                   theta_A_raw: float,
                                   theta_B_raw: float,
                                   foot_height_m: float = -0.28) -> None:
        """
        Fit encoder zero offsets such that driving position maps to a foot
        that is `foot_height_m` below the hip pivot in Z (i.e., pointing down).

        foot_height_m should be negative (foot below hip).
        X and Y of foot in Hip frame are assumed zero (straight down).

        This is a single-point calibration; refine by measuring actual foot
        position with a ruler and re-running.
        """
        target = np.array([0.0, 0.0, foot_height_m])

        def residual(offsets):
            zA, zB = offsets
            try:
                p = leg_fk(theta_A_raw, theta_B_raw, zA, zB)
            except Exception:
                return 1e6
            return float(np.sum((p - target) ** 2))

        from scipy.optimize import minimize as _min
        r = _min(residual, x0=[theta_A_raw, theta_B_raw],
                 method='Nelder-Mead',
                 options={'xatol': 1e-5, 'fatol': 1e-8, 'maxiter': 5000})
        self.zero_A, self.zero_B = float(r.x[0]), float(r.x[1])
        self._calibrated = True

    def fk(self, theta_A_raw: float, theta_B_raw: float) -> np.ndarray:
        return leg_fk(theta_A_raw, theta_B_raw, self.zero_A, self.zero_B)

    def ik(self, target: np.ndarray,
           guess_A_raw: float | None = None,
           guess_B_raw: float | None = None) -> tuple[float, float] | None:
        return leg_ik(target, self.zero_A, self.zero_B,
                      guess_A_raw, guess_B_raw)

    def foot_height(self, theta_A_raw: float, theta_B_raw: float) -> float:
        """Return Z component (vertical) of foot in Hip frame."""
        return float(self.fk(theta_A_raw, theta_B_raw)[2])


# ── Jump waypoint helper ───────────────────────────────────────────────────── #

# Driving positions from driving_leg_pos.yaml (raw encoder values)
DRIVING_POS = {
    'left':  {'A': 0.406,   'B': 2.627965, 'hip': 1.0464},
    'right': {'A': 1.7793,  'B': 1.0,      'hip': 4.9616},
}

# RS04 joint names per side
JOINT_NAMES = {
    'left':  {'hip': 'joint_rs04_1', 'A': 'joint_rs04_2', 'B': 'joint_rs04_3'},
    'right': {'hip': 'joint_rs04_4', 'A': 'joint_rs04_5', 'B': 'joint_rs04_6'},
}


def compute_jump_waypoints(
        cal_left: LegCalibration,
        cal_right: LegCalibration,
        crouch_depth: float = 0.6,
) -> dict:
    """
    Compute raw encoder positions for CROUCH and EXTEND phases of the jump.

    crouch_depth: fraction of max leg extension (0=fully retracted, 1=fully extended)
    Returns dict with keys 'crouch' and 'extend', each a dict of joint_name → angle.

    If IK fails, falls back to heuristic delta from driving position.
    """
    result = {'crouch': {}, 'extend': {}}

    for side, cal in (('left', cal_left), ('right', cal_right)):
        dp = DRIVING_POS[side]
        jn = JOINT_NAMES[side]

        # Hip stays at driving position during jump
        result['crouch'][jn['hip']] = dp['hip']
        result['extend'][jn['hip']] = dp['hip']

        crouch_target = np.array([0.0, 0.0, -L_MAX * crouch_depth])
        extend_target = np.array([0.0, 0.0, -L_MAX * 0.95])  # 95% extension (avoid singularity)

        for phase, target in (('crouch', crouch_target), ('extend', extend_target)):
            sol = cal.ik(target, guess_A_raw=dp['A'], guess_B_raw=dp['B'])
            if sol is not None:
                result[phase][jn['A']] = sol[0]
                result[phase][jn['B']] = sol[1]
            else:
                # Fallback: heuristic delta based on mechanism direction
                # Retract: decrease A, increase B; Extend: increase A, decrease B
                if phase == 'crouch':
                    result[phase][jn['A']] = dp['A'] - 0.40
                    result[phase][jn['B']] = dp['B'] + 0.40
                else:
                    result[phase][jn['A']] = dp['A'] + 0.35
                    result[phase][jn['B']] = dp['B'] - 0.35

    return result


if __name__ == '__main__':
    # Quick validation: print FK at driving positions
    cal_l = LegCalibration()
    cal_l.calibrate_from_driving_pos(
        theta_A_raw=DRIVING_POS['left']['A'],
        theta_B_raw=DRIVING_POS['left']['B'],
        foot_height_m=-0.28,
    )
    dp = DRIVING_POS['left']
    foot = cal_l.fk(dp['A'], dp['B'])
    print(f"Left driving foot pos (Hip frame): {foot}")
    print(f"  zero_A={cal_l.zero_A:.4f}  zero_B={cal_l.zero_B:.4f}")

    wps = compute_jump_waypoints(cal_l, LegCalibration())
    print(f"Crouch joints: {wps['crouch']}")
    print(f"Extend joints: {wps['extend']}")
