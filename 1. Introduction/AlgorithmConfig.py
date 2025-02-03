from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("CartPole-v1")
    .training(
        train_batch_size_per_learner=2000,
        lr=0.0004
    )
)

# 1. Manage Algorithm instance directly
algo = config.build_algo()
print(algo.train())

# 2. Run Algorithm through Ray Tune
results = tune.Tuner(
    "PPO",
    param_space=config,
    run_config=train.RunConfig(stop={"num_env_steps_sampled_lifetime": 4000}),
).fit()