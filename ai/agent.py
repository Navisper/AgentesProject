class FlappyAgent:
    def __init__(self, env):
        self.env = env              # entorno Gymnasium (Flappy Bird)
        self.model = None           # aquí se cargará el modelo RL (p.ej. PPO)

    def train(self, total_timesteps):
        """Entrena el modelo usando el método learn() de SB3."""
        self.model = PPO("MlpPolicy", self.env, verbose=1, tensorboard_log="./logs/")
        self.model.learn(total_timesteps=total_timesteps)

    def evaluate(self, episodes=10):
        """Evalúa el modelo en varios episodios, devuelve promedio de puntajes."""
        total_reward = 0
        for _ in range(episodes):
            obs, _ = self.env.reset()
            done = False
            while not done:
                action, _states = self.model.predict(obs)
                obs, reward, done, _, info = self.env.step(action)
                total_reward += reward
        return total_reward/episodes

    def save(self, path):
        """Guarda el modelo entrenado a disco."""
        if self.model:
            self.model.save(path)
