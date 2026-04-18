# mjlab_robot

RL training package for the TeamBowl two-wheeled self-balancing robot using mjlab (MuJoCo + RSL-RL / PPO).

## Key Files

| File | Purpose |
|------|---------|
| `teambowl_mjlab.xml` | Combined MJCF: teambowl_balance.xml physics + gear equality constraints + velocity actuators + sensors |
| `robot_constants.py` | MjSpec factory, EntityCfg, gear ratios, HOME_KEYFRAME, CollisionCfg |
| `scene.xml` | mjlab scene wrapper (500 Hz timestep); no floor — floor defined inside teambowl_mjlab.xml |
| `meshes/` | Symlink → `../teambowl_ws/sim/mujoco/meshes` (numbered STL files) |

## Task: TeamBowl-Balance-Flat-v0

Registered in `src/mjlab_robot/tasks/balance/__init__.py`.

- **Env config**: `src/mjlab_robot/tasks/balance/balance_env_cfg.py`
- **PPO config**: `src/mjlab_robot/tasks/balance/rl_cfg.py`
- **Episode**: 120 s at sim speed (not wall clock); 500 Hz physics / 125 Hz policy (decimation=4)
- **Envs**: 4096 training, 1 play
- **Actions**: 2 motor velocity targets in [-1,1] × scale (39.3 rad/s left, 47.1 rad/s right)
- **Obs**: 16-dim policy (projected_gravity, imu_gyro, imu_lin_vel, wheel_vel ×2, cmd ×3, last_action ×2)
- **Termination**: pitch > 0.5 rad (≈ 28°) or timeout

## XML Physics Changes from teambowl_balance.xml

- Added `<equality>` gear constraints (motor→wheel, with direction flip encoded in negative polycoef)
  - Left:  `polycoef="0 -0.08275862... 0 0 0"` (12/145 ratio)
  - Right: `polycoef="0 -0.06896551... 0 0 0"` (12/174 ratio)
- Replaced torque actuators with velocity actuators on motor joints (`act_left_motor`, `act_right_motor`)
- Added energy loss defaults: `motor_joint` class (armature, damping, frictionloss), `wheel_joint` class (armature, frictionloss)
- Added wheel/motor joint velocity sensors for observations
- Named all collision geoms (`frame_box_collision`, `left_wheel_collision`, etc.)
- Collision geom positions corrected by computing mesh centroid from STL vertices:
  - `left_wheel_collision`: `pos="0.013 0 0"` (13mm axial shift to match wheel disc center)
  - `right_wheel_collision`: `pos="-0.012 0 0"` (12mm axial shift in flipped body frame)
  - `frame_box_collision`: `pos="0 0.013 0.072"` (corrected from `0 0.05 0.1`)
- `meshdir="meshes"` in compiler; mesh refs without path prefix

## Gear Ratios

- Left motor → left wheel: N = 145/12 ≈ 12.083 (direction flips)
- Right motor → right wheel: N = 174/12 = 14.5 (direction flips)
- Max wheel surface speed: ±0.5 m/s → max motor speed ≈ ±39.3 rad/s (left), ±47.1 rad/s (right)

## venv Note

The `.venv/` Python 3.12 symlink points to a Homebrew Python that is no longer installed. Reinstall Python 3.12 via Homebrew (`brew install python@3.12`) or recreate the venv before running locally. Training is intended to run on Linux with NVIDIA GPU.

## Runner Choice

`MjlabOnPolicyRunner` (base class) is used instead of `VelocityOnPolicyRunner`. The velocity runner
hardcodes `env.action_manager.get_term("joint_pos")` and asserts `isinstance(JointPositionAction)`
in its ONNX export step — incompatible with our wheel-velocity action space. The base runner saves
`.pt` checkpoints without ONNX export, which is sufficient for this robot.

## Verification (run on Linux training machine)

```bash
# Check XML DOF counts (expect nu=2, neq=2)
python -c "
import mujoco; m = mujoco.MjModel.from_xml_path('teambowl_mjlab.xml')
print('nq:', m.nq, 'nv:', m.nv, 'nu:', m.nu, 'neq:', m.neq)
"

# Check task registers
python -c "import mjlab_robot; from mjlab.tasks.registry import list_tasks; print(list_tasks())"

# Smoke test (16 envs, 10 iters)
train --task TeamBowl-Balance-Flat-v0 --env.scene.num_envs 16 --agent.max_iterations 10

# Full training
train --task TeamBowl-Balance-Flat-v0 --env.scene.num_envs 4096 --agent.max_iterations 5000
```
