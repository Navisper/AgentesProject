import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
GAME_CODE_DIR = PROJECT_ROOT / "game" / "code"
sys.path.insert(0, str(GAME_CODE_DIR))

import gymnasium as gym
from gymnasium import spaces
import pygame
from settings import WINDOW_WIDTH, WINDOW_HEIGHT
from sprites import BG, Ground, Plane, Obstacle


class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-500, high=500, shape=(4,), dtype=float)
        self._game = None
        self.plane = None
        self.clock = None
        self.last_obstacle = None
        self.survived_frames = 0
        self._bg_path = PROJECT_ROOT / "game" / "graphics" / "environment" / "background.png"

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        if self.render_mode == "human":
            self.display_surface = pygame.display.get_surface()
        else:
            self.display_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Flappy Bird RL")
        self.clock = pygame.time.Clock()

        self.all_sprites = pygame.sprite.Group()
        self.collision_sprites = pygame.sprite.Group()

        bg_height = pygame.image.load(str(self._bg_path)).get_height()
        self.scale_factor = WINDOW_HEIGHT / bg_height

        BG(self.all_sprites, self.scale_factor)
        Ground([self.all_sprites, self.collision_sprites], self.scale_factor)
        self.plane = Plane(self.all_sprites, self.scale_factor / 1.7)

        self.obstacle_timer = pygame.USEREVENT + 1
        pygame.time.set_timer(self.obstacle_timer, 1400)

    def _reset_game(self):
        if self._game is None:
            self._init_pygame()
        else:
            self.all_sprites.empty()
            self.collision_sprites.empty()
            bg_height = pygame.image.load(str(self._bg_path)).get_height()
            self.scale_factor = WINDOW_HEIGHT / bg_height
            BG(self.all_sprites, self.scale_factor)
            Ground([self.all_sprites, self.collision_sprites], self.scale_factor)
            self.plane = Plane(self.all_sprites, self.scale_factor / 1.7)

    def _get_obs(self):
        plane_y = self.plane.rect.centery
        plane_vel = self.plane.direction
        obstacles = [o for o in self.collision_sprites if o.sprite_type == "obstacle"]
        if obstacles:
            nearest = min(obstacles, key=lambda o: o.rect.x)
            gap_y = nearest.rect.centery
            distance = nearest.rect.x - self.plane.rect.centerx
        else:
            gap_y = WINDOW_HEIGHT // 2
            distance = WINDOW_WIDTH
        return [plane_y - gap_y, plane_vel, distance, 0]

    def _check_collision(self):
        if pygame.sprite.spritecollide(self.plane, self.collision_sprites, False, pygame.sprite.collide_mask):
            return True
        if self.plane.rect.top <= 0:
            return True
        if self.plane.rect.bottom >= WINDOW_HEIGHT:
            return True
        return False

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._reset_game()
        self.survived_frames = 0
        return self._get_obs(), {}

    def step(self, action):
        dt = 1 / 60
        pygame.event.pump()

        if action == 1:
            self.plane.jump()

        self.all_sprites.update(dt)

        if pygame.sprite.spritecollide(self.plane, self.collision_sprites, False, pygame.sprite.collide_mask) \
                or self.plane.rect.top <= 0 or self.plane.rect.bottom >= WINDOW_HEIGHT:
            reward = -100
            self.plane.kill()
        else:
            reward = 1
            self.survived_frames += 1
            if self.survived_frames % 50 == 0:
                reward += 5

        done = not self.plane.alive()
        truncated = False
        info = {"score": self.survived_frames}

        if self.render_mode == "human":
            self.display_surface.fill((135, 206, 235))
            self.all_sprites.draw(self.display_surface)
            pygame.display.flip()
            self.clock.tick(60)

        return self._get_obs(), reward, done, truncated, info

    def close(self):
        if self._game is not None:
            pygame.quit()
            self._game = None