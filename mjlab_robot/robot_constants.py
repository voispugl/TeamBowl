from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

robot_xml: Path = (
  "robot.xml"
)
assert robot_xml.exists()




def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, robot_xml.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(robot_xml))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Motor specs (from Unitree).
ROTOR_INERTIA_04 = 1.1e-4,


GEARS_04 = 9
ARMATURE_04 = reflected_inertia(
  ROTOR_INERTIA_04, GEARS_04
)

ROTOR_INERTIA_00 = 0.139e-4,

GEARS_00 = 10
ARMATURE_00 = reflected_inertia(
  ROTOR_INERTIA_00, GEARS_00
)

ROTOR_INERTIA_05 = 0.068e-4,
GEARS_05 = 7.75

ARMATURE_05 = reflected_inertia(
  ROTOR_INERTIA_05, GEARS_05
)

ROTOR_INERTIA_vesc = 0.6e-4
GEARS_vesc = 1

ARMATURE_vesc = reflected_inertia(
  ROTOR_INERTIA_vesc, GEARS_vesc
)

ACTUATOR_04 = ElectricActuator(
  reflected_inertia=ARMATURE_04,
  velocity_limit=10.0,
  effort_limit=120.0,
)
ACTUATOR_00 = ElectricActuator(
  reflected_inertia=ARMATURE_00,
  velocity_limit=10.0,
  effort_limit=14.0,
)
ACTUATOR_05 = ElectricActuator(
  reflected_inertia=ARMATURE_05,
  velocity_limit=10.0,
  effort_limit=5.5,
)
ACTUATOR_vesc = ElectricActuator(
  reflected_inertia=ARMATURE_vesc,
  velocity_limit=315.0,
  effort_limit=1.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_04 = ARMATURE_04 * NATURAL_FREQ**2
STIFFNESS_00 = ARMATURE_00 * NATURAL_FREQ**2
STIFFNESS_05 = ARMATURE_05 * NATURAL_FREQ**2
STIFFNESS_vesc = ARMATURE_vesc * NATURAL_FREQ**2

DAMPING_04 = 2.0 * DAMPING_RATIO * ARMATURE_04 * NATURAL_FREQ
DAMPING_00 = 2.0 * DAMPING_RATIO * ARMATURE_00 * NATURAL_FREQ
DAMPING_05 = 2.0 * DAMPING_RATIO * ARMATURE_05 * NATURAL_FREQ
DAMPING_vesc = 2.0 * DAMPING_RATIO * ARMATURE_vesc * NATURAL_FREQ

ROBOT_ACTUATOR_04 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_04,
  damping=DAMPING_04,
  effort_limit=ACTUATOR_04.effort_limit,
  armature=ACTUATOR_04.reflected_inertia,
)
ROBOT_ACTUATOR_00 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_00,
  damping=DAMPING_00,
  effort_limit=ACTUATOR_00.effort_limit,
  armature=ACTUATOR_00.reflected_inertia,
)
ROBOT_ACTUATOR_05 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_05,
  damping=DAMPING_05,
  effort_limit=ACTUATOR_05.effort_limit,
  armature=ACTUATOR_05.reflected_inertia,
)
ROBOT_ACTUATOR_vesc = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=STIFFNESS_vesc,
  damping=DAMPING_vesc,
  effort_limit=ACTUATOR_vesc.effort_limit,
  armature=ACTUATOR_vesc.reflected_inertia,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.783675),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

ROBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ROBOT_ACTUATOR_04,
    ROBOT_ACTUATOR_00,
    ROBOT_ACTUATOR_05,
    ROBOT_ACTUATOR_vesc,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_robot_cfg() -> EntityCfg:
  """Get a fresh robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=ROBOT_ARTICULATION,
  )


ROBOT_ACTION_SCALE: dict[str, float] = {}
for a in ROBOT_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    ROBOT_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_robot_cfg())

  viewer.launch(robot.spec.compile())

