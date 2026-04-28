from stable_baselines3 import PPO


class FlappyAgent:
    def __init__(self, env):
        self.env = env
        self.model = None

    def train(self, total_timesteps, save_path="ai/checkpoints/ppo_flappy"):
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log="ai/logs/",
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=10,
            gamma=0.995,
            ent_coef=0.01,
        )
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
        self.save(save_path)

    def evaluate(self, episodes=10):
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")
        total_reward = 0
        for _ in range(episodes):
            obs, _ = self.env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = self.env.step(action)
                total_reward += reward
        return total_reward / episodes

    def save(self, path):
        if self.model:
            self.model.save(path)

    def load(self, path):
        if self.model:
            self.model = PPO.load(path, env=self.env)