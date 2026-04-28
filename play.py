from ai.flappy_env import FlappyBirdEnv
from stable_baselines3 import PPO

env = FlappyBirdEnv(render_mode="human")
model = PPO.load("ai/checkpoints/ppo_flappy", env=env)

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

env.close()
print("Done! Check the game window.")
