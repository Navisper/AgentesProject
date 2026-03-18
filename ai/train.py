import gymnasium as gym
from agent import FlappyAgent
from stable_baselines3 import PPO

# Crear entorno (por ejemplo, un entorno personalizado de Flappy Bird)
env = gym.make("FlappyBird-v0")  # nombre hipotético de entorno
agent = FlappyAgent(env)

# Entrenar agente
agent.train(total_timesteps=100_000)
agent.save("ai/checkpoints/ppo_flappy")
