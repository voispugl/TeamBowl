"""TeamBowl balance task — registers TeamBowl-Balance-Flat-v0."""

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .balance_env_cfg import teambowl_balance_flat_env_cfg
from .rl_cfg import teambowl_balance_ppo_runner_cfg

register_mjlab_task(
    task_id="TeamBowl-Balance-Flat-v0",
    env_cfg=teambowl_balance_flat_env_cfg(),
    play_env_cfg=teambowl_balance_flat_env_cfg(play=True),
    rl_cfg=teambowl_balance_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)
