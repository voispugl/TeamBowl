"""Balance RL environment configuration for the TeamBowl two-wheeled robot.

Control pipeline mirrored from ROS2:
  Planner → /cmd_vel_auto (Twist: linear.x, angular.z)
        → vel_cmd_mux / collision_guard → /cmd_vel_safe
        → balance_controller

Observation space (16 dims, policy):
  projected_gravity (3)  — gravity direction in body frame → encodes pitch/roll
  base_ang_vel (3)       — body angular velocity (pitch_rate, roll_rate, yaw_rate)
  base_lin_vel (3)       — body forward/lateral/vertical velocity
  left_wheel_vel (1)     — left wheel angular velocity (rad/s)
  right_wheel_vel (1)    — right wheel angular velocity (rad/s)
  command (3)            — desired [vx, vy, wz] velocity from planner
  last_action (2)        — previous motor velocity targets

Action space (2 dims):
  [act_left_motor, act_right_motor] ∈ [-1, 1] × scale (rad/s)
  Left  scale ≈ 39.3 rad/s  (= 0.5 m/s / wheel_radius / (1/N_left))
  Right scale ≈ 47.1 rad/s  (= 0.5 m/s / wheel_radius / (1/N_right))

Domain randomisation (per-reset):
  - Frame mass          ±30 %           (body_mass)
  - Frame CoM           ±0.18 m (x/y/z) (body_ipos)
  - Frame inertia       ±30 %           (body_inertia)
  - Wheel mass          ±30 %           (body_mass, Wheel + Wheel_1)
  - Wheel inertia       ±30 %           (body_inertia)
  - Motor joint damping ±30 %           (dof_damping)
  - Motor frictionloss  ±30 %           (dof_frictionloss)
  - Wheel frictionloss  ±30 %           (dof_frictionloss)
  - Motor armature      ±30 %           (dof_armature)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make robot_constants.py accessible as a top-level import.
# robot_constants.py lives at mjlab_robot/ (parent of src/).
# ---------------------------------------------------------------------------
_robot_root = Path(__file__).parent.parent.parent.parent.parent  # mjlab_robot/
if str(_robot_root) not in sys.path:
    sys.path.insert(0, str(_robot_root))

from robot_constants import (  # noqa: E402
    GEAR_RATIO_LEFT,
    GEAR_RATIO_RIGHT,
    MAX_MOTOR_SPEED_LEFT,
    MAX_MOTOR_SPEED_RIGHT,
    WHEEL_RADIUS,
    get_robot_cfg,
)

from mjlab.envs import ManagerBasedRlEnvCfg, mdp as envs_mdp
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Robot body size ≈ 2 ft = 0.6096 m in all directions; ±30% CG shift = ±0.18 m
_ROBOT_SIZE_M: float = 0.6096
_CG_SHIFT_M: float = 0.30 * _ROBOT_SIZE_M  # 0.183 m

# Velocity command limits (matched to ROS2 collision_guard clamps)
_MAX_LIN_VEL: float = 0.15   # m/s
_MAX_ANG_VEL: float = 0.40   # rad/s

# Episode: 2 minutes at sim speed (not wall clock).
_EPISODE_LENGTH_S: float = 120.0

# Physics: 500 Hz; policy runs every 4 physics steps → 125 Hz control.
_TIMESTEP: float = 0.002
_DECIMATION: int = 4


# ---------------------------------------------------------------------------
# Environment config factory
# ---------------------------------------------------------------------------

def teambowl_balance_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the flat-terrain balance RL env config for the TeamBowl robot.

    Args:
        play: If True, uses 1 env and disables observation noise + DR events.
    """
    num_envs = 1 if play else 4096

    # ── Scene ────────────────────────────────────────────────────────────────
    scene = SceneCfg(
        num_envs=num_envs,
        env_spacing=8.0,
        terrain=TerrainImporterCfg(
            terrain_type="plane",
            env_spacing=8.0,
            num_envs=num_envs,
        ),
        entities={"robot": get_robot_cfg()},
    )

    # ── Observations ─────────────────────────────────────────────────────────
    # Sensor name format: "entity_name/xml_sensor_name"
    # All sensor names match those in teambowl_mjlab.xml.

    def _unoise(magnitude: float) -> Unoise | None:
        """Return a uniform noise term, or None in play mode (no corruption)."""
        if play:
            return None
        return Unoise(n_min=-magnitude, n_max=magnitude)

    policy_terms: dict[str, ObservationTermCfg] = {
        # Gravity vector projected into body frame (3D).
        # When upright: [0, 0, -1].  Encodes pitch + roll.
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            noise=_unoise(0.05),
        ),
        # Body-frame angular velocity [wx, wy, wz] in rad/s from imu_gyro sensor.
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_gyro"},
            noise=_unoise(0.2),
        ),
        # Body-frame linear velocity [vx, vy, vz] in m/s from imu_lin_vel sensor.
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=_unoise(0.3),
        ),
        # Left wheel angular velocity (rad/s) — from jointvel sensor.
        "left_wheel_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/left_wheel_vel"},
            noise=_unoise(0.5),
        ),
        # Right wheel angular velocity (rad/s) — from jointvel sensor.
        "right_wheel_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/right_wheel_vel"},
            noise=_unoise(0.5),
        ),
        # Velocity command from planner: [vx_cmd, vy_cmd, wz_cmd] in m/s / rad/s.
        # vy_cmd is always 0 for a wheeled robot; the policy learns to ignore it.
        "command": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": "twist"},
        ),
        # Previous motor velocity actions (2 values: left, right motor rad/s).
        "last_action": ObservationTermCfg(
            func=envs_mdp.last_action,
        ),
    }

    # Critic gets ground-truth sensors for better value estimation.
    critic_terms: dict[str, ObservationTermCfg] = {
        **policy_terms,
        "gt_linvel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/gt_linvel"},
        ),
        "gt_angvel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/gt_angvel"},
        ),
    }

    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # ── Actions ──────────────────────────────────────────────────────────────
    # Policy outputs ∈ [-1, 1]; scale maps to motor angular velocity (rad/s).
    # The equality constraint in the XML then drives the wheel at 1/N × motor speed.
    #   Left:  scale = 0.5 m/s / wheel_radius * N_left  ≈ 39.3 rad/s
    #   Right: scale = 0.5 m/s / wheel_radius * N_right ≈ 47.1 rad/s
    actions: dict[str, ActionTermCfg] = {
        # actuator_names are matched against the JOINT names of actuated joints.
        # Our XML <velocity> actuators target left_motor_0 / right_motor_0, so
        # those joint names are what JointVelocityActionCfg resolves against.
        "motor_vel": JointVelocityActionCfg(
            entity_name="robot",
            actuator_names=("left_motor_0", "right_motor_0"),
            scale={
                "left_motor_0":  MAX_MOTOR_SPEED_LEFT,   # ≈ 39.3 rad/s
                "right_motor_0": MAX_MOTOR_SPEED_RIGHT,  # ≈ 47.1 rad/s
            },
            use_default_offset=False,
        ),
    }

    # ── Commands ─────────────────────────────────────────────────────────────
    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(8.0, 20.0),    # re-sample every 8–20 s
            rel_standing_envs=0.15,                # 15 % of envs: balance in place
            rel_heading_envs=0.0,                  # wheeled robot: no heading control
            heading_command=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-_MAX_LIN_VEL, _MAX_LIN_VEL),
                lin_vel_y=(0.0, 0.0),              # no lateral motion
                ang_vel_z=(-_MAX_ANG_VEL, _MAX_ANG_VEL),
                heading=None,
            ),
        ),
    }

    # ── Rewards ──────────────────────────────────────────────────────────────
    rewards: dict[str, RewardTermCfg] = {
        # Stay upright: rewards small xy-component of projected gravity.
        # exp(-xy_grav² / std²); std=0.15 → ≈63% at 15° tilt, ≈0% at 45°.
        # Must specify body_names="Frame" so flat_orientation operates on the
        # single root body (body_ids=slice(None) selects all bodies by default,
        # causing a shape mismatch in quat_apply_inverse).
        "upright": RewardTermCfg(
            func=vel_mdp.flat_orientation,
            params={"std": 0.15, "asset_cfg": SceneEntityCfg("robot", body_names="Frame")},
            weight=3.0,
        ),
        # Survive reward: +1 every step while not fallen.
        "alive": RewardTermCfg(
            func=envs_mdp.is_alive,
            weight=0.5,
        ),
        # Terminal penalty: negative reward on fall (not on timeout).
        "fall_penalty": RewardTermCfg(
            func=envs_mdp.is_terminated,
            weight=-2.0,
        ),
        # Forward velocity tracking — exp(-error²/std²).
        "track_lin_vel": RewardTermCfg(
            func=vel_mdp.track_linear_velocity,
            params={"std": 0.25, "command_name": "twist"},
            weight=2.0,
        ),
        # Yaw rate tracking.
        "track_ang_vel": RewardTermCfg(
            func=vel_mdp.track_angular_velocity,
            params={"std": 0.25, "command_name": "twist"},
            weight=1.0,
        ),
        # Penalise sudden changes in motor velocity command → smoother riding.
        "action_smoothness": RewardTermCfg(
            func=envs_mdp.action_rate_l2,
            weight=-0.01,
        ),
        # Penalise motor velocity magnitude (energy efficiency).
        # Uses the motor joints (not wheel joints) since that's what draws current.
        "motor_energy": RewardTermCfg(
            func=envs_mdp.joint_vel_l2,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=("left_motor_0", "right_motor_0")
                )
            },
            weight=-0.0005,
        ),
    }

    # ── Terminations ─────────────────────────────────────────────────────────
    terminations: dict[str, TerminationTermCfg] = {
        # Fall: tilt > 0.5 rad (≈28°) — matches real robot estop threshold.
        "fall": TerminationTermCfg(
            func=envs_mdp.bad_orientation,
            params={"limit_angle": 0.5},
        ),
        # Episode timeout (non-terminal, triggers reset but no terminal penalty).
        "time_out": TerminationTermCfg(
            func=envs_mdp.time_out,
            time_out=True,
        ),
    }

    # ── Domain Randomisation Events ──────────────────────────────────────────
    if play:
        # In play mode: only reset position, no DR.
        events: dict[str, EventTermCfg] = {
            "reset_base": EventTermCfg(
                func=envs_mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (0.0, 0.02),
                        "pitch": (-0.05, 0.05),
                        "yaw": (-math.pi, math.pi),
                    },
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
        }
    else:
        events = {
            # Reset base position + small pitch perturbation (forces recovery learning).
            "reset_base": EventTermCfg(
                func=envs_mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (0.0, 0.02),
                        "roll":  (-0.05, 0.05),
                        "pitch": (-0.08, 0.08),   # random lean → forces recovery
                        "yaw":   (-math.pi, math.pi),
                    },
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),

            # ── Frame (main body) mass ±30 % ─────────────────────────────
            "rand_frame_mass": EventTermCfg(
                func=envs_mdp.randomize_field,
                mode="reset",
                params={
                    "field": "body_mass",
                    "ranges": (0.70, 1.30),
                    "operation": "scale",
                    "asset_cfg": SceneEntityCfg("robot", body_names=("Frame",)),
                },
            ),

            # ── Frame CoM shift ±30 % of 0.6 m (= ±0.183 m) in x/y/z ───
            "rand_frame_com": EventTermCfg(
                func=envs_mdp.randomize_field,
                mode="reset",
                params={
                    "field": "body_ipos",            # CoM offset in body frame
                    "ranges": (-_CG_SHIFT_M, _CG_SHIFT_M),
                    "operation": "add",
                    "asset_cfg": SceneEntityCfg("robot", body_names=("Frame",)),
                },
            ),

            # ── Frame inertia ±30 % ───────────────────────────────────────
            "rand_frame_inertia": EventTermCfg(
                func=envs_mdp.randomize_field,
                mode="reset",
                params={
                    "field": "body_inertia",
                    "ranges": (0.70, 1.30),
                    "operation": "scale",
                    "asset_cfg": SceneEntityCfg("robot", body_names=("Frame",)),
                },
            ),

            # ── Wheel mass ±30 % ─────────────────────────────────────────
            # body_names uses re.fullmatch; "Wheel" matches exactly "Wheel",
            # "Wheel_1" matches exactly "Wheel_1" (not "Wheel Gear" etc.).
            "rand_wheel_mass": EventTermCfg(
                func=envs_mdp.randomize_field,
                mode="reset",
                params={
                    "field": "body_mass",
                    "ranges": (0.70, 1.30),
                    "operation": "scale",
                    "asset_cfg": SceneEntityCfg(
                        "robot", body_names=("Wheel", "Wheel_1")
                    ),
                },
            ),

            # ── Wheel inertia ±30 % ──────────────────────────────────────
            "rand_wheel_inertia": EventTermCfg(
                func=envs_mdp.randomize_field,
                mode="reset",
                params={
                    "field": "body_inertia",
                    "ranges": (0.70, 1.30),
                    "operation": "scale",
                    "asset_cfg": SceneEntityCfg(
                        "robot", body_names=("Wheel", "Wheel_1")
                    ),
                },
            ),

            # ── Motor joint damping (back-EMF model) ±30 % ───────────────
            "rand_motor_damping": EventTermCfg(
                func=envs_mdp.randomize_field,
                mode="reset",
                params={
                    "field": "dof_damping",
                    "ranges": (0.70, 1.30),
                    "operation": "scale",
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=("left_motor_0", "right_motor_0"),
                    ),
                },
            ),

            # ── Motor armature (reflected rotor inertia) ±30 % ───────────
            "rand_motor_armature": EventTermCfg(
                func=envs_mdp.randomize_field,
                mode="reset",
                params={
                    "field": "dof_armature",
                    "ranges": (0.70, 1.30),
                    "operation": "scale",
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=("left_motor_0", "right_motor_0"),
                    ),
                },
            ),

            # ── Motor and wheel joint friction ±30 % ─────────────────────
            "rand_drivetrain_friction": EventTermCfg(
                func=envs_mdp.randomize_field,
                mode="reset",
                params={
                    "field": "dof_frictionloss",
                    "ranges": (0.70, 1.30),
                    "operation": "scale",
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=(
                            "left_motor_0",
                            "right_motor_0",
                            "left_wheel_0",
                            "right_wheel_0",
                        ),
                    ),
                },
            ),
        }

    # ── Simulation config ────────────────────────────────────────────────────
    sim = SimulationCfg(
        mujoco=MujocoCfg(timestep=_TIMESTEP),   # 500 Hz physics
    )

    # ── Assemble and return ──────────────────────────────────────────────────
    return ManagerBasedRlEnvCfg(
        decimation=_DECIMATION,            # policy runs at 500/4 = 125 Hz
        scene=scene,
        observations=observations,
        actions=actions,
        commands=commands,
        rewards=rewards,
        terminations=terminations,
        events=events,
        sim=sim,
        episode_length_s=_EPISODE_LENGTH_S,  # 120 s (runs at sim speed)
    )
