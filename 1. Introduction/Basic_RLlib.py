from pprint import pprint
import ray
from ray.rllib.algorithms.ppo import PPOConfig



# Ray 초기화
ray.init(ignore_reinit_error=True)

# PPO 알고리즘 설정
config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment("CartPole-v1") # 학습 환경
    .env_runners(num_env_runners=1) # 병렬 실행할 환경 개수
)

# PPO 알고리즘 객체 생성
algo = config.build()

# 학습 실행
for i in range(10):
    result = algo.train()

    # 평균 보상
    avg_reward = result["env_runners"].get("episode_return_mean", "N/A")
    print(f"Iteration {i+1} | 평균 보상: {avg_reward}")

    # 체크포인트 저장
    if i % 5 == 0:
        checkpoint_dir = algo.save_to_path()
        print(f"Checkpoint saved in directory: {checkpoint_dir}")

# Ray 종료
ray.shutdown()