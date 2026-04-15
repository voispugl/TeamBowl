"""PPO runner configuration for the TeamBowl balance task."""

from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


def teambowl_balance_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create PPO runner configuration for the TeamBowl balance task."""
    return RslRlOnPolicyRunnerCfg(
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=0.5,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=(256, 128, 64),
            critic_hidden_dims=(256, 128, 64),
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="teambowl_balance",
        save_interval=200,
        # 48 steps × 125 Hz control = ~0.38 s rollout per update.
        # Longer rollouts help with the 120 s episode horizon.
        num_steps_per_env=48,
        max_iterations=5000,
        seed=42,
        logger="tensorboard",
    )
