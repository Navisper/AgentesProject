from ai.flappy_env import FlappyBirdEnv
from ai.agent import FlappyAgent


def main():
    env = FlappyBirdEnv(render_mode=None)
    agent = FlappyAgent(env)

    print("Starting training...")
    agent.train(total_timesteps=2_000_000, save_path="ai/checkpoints/ppo_flappy")

    print("\nEvaluating...")
    avg_reward = agent.evaluate(episodes=5)
    print(f"Average reward: {avg_reward:.2f}")


if __name__ == "__main__":
    main()