import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
GAME_CODE_DIR = PROJECT_ROOT / "game" / "code"
sys.path.insert(0, str(GAME_CODE_DIR))

import gymnasium as gym
from gymnasium import spaces
import pygame
from settings import WINDOW_WIDTH, WINDOW_HEIGHT, FRAMERATE
from sprites import BG, Ground, Plane, Obstacle


class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FRAMERATE}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(2)

        # 5 features:
        #   0: plane_y / WINDOW_HEIGHT          (normalized vertical position)
        #   1: plane_vel / 600                  (normalized velocity, jump = -450)
        #   2: obstacle_distance / WINDOW_WIDTH  (normalized distance to nearest obstacle ahead)
        #   3: obstacle_center_y / WINDOW_HEIGHT (normalized y of nearest obstacle)
        #   4: obstacle_type (-1=bottom-rising, 0=no obstacle, +1=top-hanging)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(5,), dtype=float
        )

        self._game_dir = PROJECT_ROOT / "game"
        self._bg_path = self._game_dir / "graphics" / "environment" / "background.png"

        # Obstacle spawning (time-based, no pygame events needed)
        self._obstacle_interval = 0.6  # seconds between obstacle spawns
        self._next_obstacle_time = 0.0
        self._elapsed_time = 0.0

        # Lazy-initialized in first reset
        self.plane = None
        self.display_surface = None
        self.clock = None
        self.scale_factor = None
        self.survived_frames = 0
        self._passed_obstacles = set()
        self._init_done = False

    # ── pygame lifecycle ──────────────────────────────────────────────

    def _init_pygame(self):
        """One-time pygame initialization."""
        if self._init_done:
            return

        pygame.init()
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

        if self.render_mode == "human":
            pygame.display.set_caption("Flappy Bird RL")

        self.clock = pygame.time.Clock()

        self.all_sprites = pygame.sprite.Group()
        self.collision_sprites = pygame.sprite.Group()

        bg_height = pygame.image.load(str(self._bg_path)).get_height()
        self.scale_factor = WINDOW_HEIGHT / bg_height

        BG(self.all_sprites, self.scale_factor)
        Ground([self.all_sprites, self.collision_sprites], self.scale_factor)
        self.plane = Plane(
            self.all_sprites,
            self.scale_factor / 1.7,
            gravity=450,
            jump_strength=-450,
        )

        # Tune obstacle speed for playability
        Obstacle.speed = 320

        self._init_done = True

    def _reset_scene(self):
        """Clears and re-creates all sprites for a new episode."""
        if not self._init_done:
            self._init_pygame()
            return

        self.all_sprites.empty()
        self.collision_sprites.empty()

        bg_height = pygame.image.load(str(self._bg_path)).get_height()
        self.scale_factor = WINDOW_HEIGHT / bg_height

        BG(self.all_sprites, self.scale_factor)
        Ground([self.all_sprites, self.collision_sprites], self.scale_factor)
        self.plane = Plane(
            self.all_sprites,
            self.scale_factor / 1.7,
            gravity=450,
            jump_strength=-450,
        )

    # ── observation ───────────────────────────────────────────────────

    def _get_obs(self):
        plane_y_norm = self.plane.rect.centery / WINDOW_HEIGHT
        plane_vel_norm = self.plane.direction / 600.0

        obstacles = [o for o in self.collision_sprites if o.sprite_type == "obstacle"]
        obstacles_ahead = [o for o in obstacles if o.rect.right > self.plane.rect.left]

        if obstacles_ahead:
            nearest = min(obstacles_ahead, key=lambda o: o.rect.x)
            obstacle_dist_norm = (nearest.rect.centerx - self.plane.rect.centerx) / WINDOW_WIDTH

            # Normalized y of the obstacle centre
            obstacle_y_norm = nearest.rect.centery / WINDOW_HEIGHT

            # Type: +1 → top-hanging (plane must stay below), -1 → bottom-rising (plane must stay above)
            if nearest.rect.top <= 0:
                obstacle_type = 1.0
            elif nearest.rect.bottom >= WINDOW_HEIGHT:
                obstacle_type = -1.0
            else:
                obstacle_type = 0.0
        else:
            obstacle_dist_norm = 1.0
            obstacle_y_norm = 0.5
            obstacle_type = 0.0

        return [
            plane_y_norm,
            plane_vel_norm,
            obstacle_dist_norm,
            obstacle_y_norm,
            obstacle_type,
        ]

    # ── collision ─────────────────────────────────────────────────────

    def _check_collision(self):
        if not self.plane.alive():
            return True
        if pygame.sprite.spritecollide(
            self.plane,
            self.collision_sprites,
            False,
            pygame.sprite.collide_mask,
        ):
            return True
        if self.plane.rect.top <= 0:
            return True
        if self.plane.rect.bottom >= WINDOW_HEIGHT:
            return True
        return False

    # ── scoring ───────────────────────────────────────────────────────

    def _count_passed_obstacles(self):
        """Returns how many *new* obstacles have passed behind the plane this step."""
        count = 0
        for sprite in self.collision_sprites:
            if sprite.sprite_type != "obstacle":
                continue
            if sprite.rect.right >= self.plane.rect.left:
                continue
            if id(sprite) in self._passed_obstacles:
                continue
            self._passed_obstacles.add(id(sprite))
            count += 1
        return count

    # ── Gym API ───────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._reset_scene()
        self.survived_frames = 0
        self._passed_obstacles.clear()
        self._elapsed_time = 0.0
        self._next_obstacle_time = self._obstacle_interval
        return self._get_obs(), {}

    def step(self, action):
        dt = 1.0 / FRAMERATE  # physics step locked to 120 Hz

        # ── time-based obstacle spawning (no pygame events needed) ──
        self._elapsed_time += dt
        while self._elapsed_time >= self._next_obstacle_time:
            Obstacle(
                [self.all_sprites, self.collision_sprites],
                self.scale_factor * 1.1,
            )
            self._next_obstacle_time += self._obstacle_interval

        # ── agent action ─────────────────────────────────────────────
        if action == 1:
            self.plane.jump()

        # ── physics update ───────────────────────────────────────────
        self.all_sprites.update(dt)

        # ── collision & reward ───────────────────────────────────────
        dead = self._check_collision()
        obstacles_passed = self._count_passed_obstacles()

        if dead:
            reward = -10.0
            self.plane.kill()
        else:
            # Small survival incentive + obstacle bonus
            reward = 0.1 + obstacles_passed * 10.0
            self.survived_frames += 1

        done = dead
        truncated = False
        info = {"score": self.survived_frames}

        # ── rendering ────────────────────────────────────────────────
        if self.render_mode == "human":
            self.display_surface.fill((135, 206, 235))
            self.all_sprites.draw(self.display_surface)
            pygame.display.flip()
            self.clock.tick(FRAMERATE)

        return self._get_obs(), reward, done, truncated, info

    def close(self):
        pygame.quit()
        self._init_done = False
