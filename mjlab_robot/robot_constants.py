"""TeamBowl robot constants for mjlab.

This module defines the MuJoCo spec, assets, initial-state keyframe,
collision config, and entity config for the TeamBowl two-wheeled
self-balancing robot.

Robot: two-wheeled balancer with two rigid (welded) legs.
MJCF:  teambowl_mjlab.xml   (based on teambowl_balance.xml)
Meshes: mjlab_robot/meshes/ → symlink to teambowl_ws/sim/mujoco/meshes/

Gearing (from onshape-to-robot equality constraints):
  Left  wheel: wheel_angle = -(12/145) * motor_angle  →  N_left  = 145/12 ≈ 12.08
  Right wheel: wheel_angle = -(12/174) * motor_angle  →  N_right = 174/12 = 14.50
  Direction flip encoded by negative polycoef.

Motor model (VESC hub motors):
  Rotor inertia  I_rotor = 0.6e-4 kg·m²
  Armature left  = I_rotor * N_left²  = 0.6e-4 * (145/12)²  ≈ 8.76e-3 kg·m²
  Armature right = I_rotor * N_right² = 0.6e-4 * (174/12)²  ≈ 12.6e-3 kg·m²
  Damping (back-EMF proxy) = 0.05 N·m·s/rad   (set in XML default class)
  Frictionloss (Coulomb)   = 0.10 N·m          (set in XML default class)

Initial height:
  Wheel_1 body sits at local z = -0.056195 from Frame.
  Wheel radius = 0.154 m.  Floor at z = -0.3 (from XML).
  Frame z for wheels touching floor: -0.3 + 0.154 + 0.056195 = -0.0898 ≈ -0.090 m
"""

from dataclasses import dataclass, field
from pathlib import Path

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

# Absolute path so this works regardless of working directory.
robot_xml: Path = Path(__file__).parent / "teambowl_mjlab.xml"
assert robot_xml.exists(), f"TeamBowl MJCF not found: {robot_xml}"

_meshes_dir: Path = Path(__file__).parent / "meshes"
assert _meshes_dir.exists(), (
    f"Mesh directory not found: {_meshes_dir}\n"
    "Create the symlink:  cd mjlab_robot && ln -s ../teambowl_ws/sim/mujoco/meshes meshes"
)


def get_assets(meshdir: str) -> dict[str, bytes]:
    """Load all STL mesh files from the meshes/ directory."""
    assets: dict[str, bytes] = {}
    update_assets(assets, _meshes_dir, meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    """Return a compiled MjSpec for the TeamBowl robot."""
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Gear / actuator constants (for reference in env config).
##

#: Left-side gear ratio (motor turns N times for each wheel revolution)
GEAR_RATIO_LEFT: float = 145.0 / 12.0      # ≈ 12.083

#: Right-side gear ratio
GEAR_RATIO_RIGHT: float = 174.0 / 12.0     # = 14.5

#: Wheel radius (metres) — from XML cylinder collision geom size
WHEEL_RADIUS: float = 0.154

#: Max wheel surface speed (m/s) matching ROS2 collision_guard limits
MAX_WHEEL_SURFACE_SPEED_MS: float = 0.5

#: Corresponding max motor angular speed (rad/s) for each side
MAX_MOTOR_SPEED_LEFT: float = MAX_WHEEL_SURFACE_SPEED_MS / WHEEL_RADIUS * GEAR_RATIO_LEFT   # ≈ 39.3
MAX_MOTOR_SPEED_RIGHT: float = MAX_WHEEL_SURFACE_SPEED_MS / WHEEL_RADIUS * GEAR_RATIO_RIGHT  # ≈ 47.1


##
# Initial-state keyframe.
##

# Frame body z so wheels (local z ≈ -0.056 m from Frame, radius 0.154 m)
# rest on the floor (z = -0.3 m in teambowl_mjlab.xml):
#   Frame_z = floor_z + wheel_radius + |wheel_local_z|
#           = -0.3     + 0.154       + 0.056195       = -0.0898 m
_SPAWN_Z: float = -0.090

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, _SPAWN_Z),
    rot=(1.0, 0.0, 0.0, 0.0),  # identity — robot stands upright
    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    joint_pos={},               # no controllable joints; legs are rigid/welded
    joint_vel={".*": 0.0},
)


##
# Collision config.
##

# Geoms named in teambowl_mjlab.xml:
#   frame_box_collision      — main body box (invisible)
#   left_wheel_collision     — left wheel cylinder
#   right_wheel_collision    — right wheel cylinder
#   left_foot_collision      — left foot mesh
#   right_foot_collision     — right foot mesh
#
# Feet get condim=3 (full friction pyramid) with higher priority so wheel/foot
# contacts don't interfere.  Everything else gets condim=1 (frictionless normal).

WHEEL_FOOT_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={
        r"^(left|right)_foot_collision$": 3,
        ".*_collision": 1,
    },
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (0.8,)},
)


##
# Articulation config.
##

# Velocity actuators are defined entirely in the XML (teambowl_mjlab.xml).
# mjlab does not need to manage them through BuiltinActuatorCfg — pass an
# empty actuator tuple so mjlab leaves the XML actuators untouched.
ROBOT_ARTICULATION = EntityArticulationInfoCfg(actuators=())


##
# Final entity config.
##

def get_robot_cfg() -> EntityCfg:
    """Return a fresh TeamBowl EntityCfg instance.

    Returns a new instance each call to avoid mutation issues when the same
    config object is shared across multiple environment instances.
    """
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(WHEEL_FOOT_COLLISION,),
        spec_fn=get_spec,
        articulation=ROBOT_ARTICULATION,
    )


##
# Dev/debug entry point — opens the MuJoCo passive viewer.
##

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    print(f"Loading: {robot_xml}")
    robot_cfg = get_robot_cfg()
    robot = Entity(robot_cfg)
    compiled = robot.spec.compile()
    print(
        f"  nq={compiled.nq}  nv={compiled.nv}  nu={compiled.nu}"
        f"  neq={compiled.neq}  nsensor={compiled.nsensor}"
    )
    print("Opening MuJoCo viewer...")
    viewer.launch(compiled)
