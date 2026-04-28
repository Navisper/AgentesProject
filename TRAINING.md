# Entrenamiento del agente Flappy Bird con PPO

## Resumen de cambios realizados

### `ai/flappy_env.py` (reescrito)
| Antes (bug) | Después (corregido) |
|---|---|
| Los obstáculos **nunca se generaban** — el timer de pygame se configuraba pero `step()` no procesaba los eventos | `step()` consume el event queue con `pygame.event.get()` y spawnea obstáculos cuando el timer dispara (`flappy_env.py:180`) |
| `dt` fijo en `1/60` — física inconsistente con el juego real | `dt = 1.0 / FRAMERATE = 1/120`, igual que el juego original (`flappy_env.py:177`) |
| Observación de 4 features, la 4ª siempre `0` — espacio inútil | Observación de 5 features normalizados: `[plane_y_norm, plane_vel_norm, obstacle_dist_norm, obstacle_y_norm, obstacle_type]` (`flappy_env.py:99-131`) |
| `_check_collision()` definido pero **nunca usado** — código muerto | Se usa correctamente en `step()` para detectar colisiones (`flappy_env.py:195`) |
| Sin recompensa por esquivar obstáculos | `+10` por cada obstáculo superado, trackeado con un set para evitar double-counting (`flappy_env.py:153-165`) |
| Headless mode creaba una ventana innecesaria | Si `render_mode=None`, usa `pygame.Surface` sin display window (`flappy_env.py:60-61`) |

### `game/code/sprites.py`
| Antes | Después |
|---|---|
| `BG.__init__` llamaba `pygame.display.set_mode()` — un sprite no debería crear la ventana | Línea eliminada; la responsabilidad es del `Game` o del `Env` |

---

## Requisitos previos

```bash
pip install gymnasium stable-baselines3 pygame tensorboard
```

---

## Cómo entrenar la IA

Ejecuta desde la raíz del proyecto (donde está `main.py`):

```bash
python -m ai.train
```

Esto hace lo siguiente:
1. Crea el entorno Flappy Bird en modo headless (sin ventana)
2. Entrena un agente PPO durante **500,000 timesteps**
3. Guarda el checkpoint en `ai/checkpoints/ppo_flappy.zip`
4. Evalúa el agente en 5 episodios y muestra la recompensa promedio

### Monitorear el entrenamiento con TensorBoard

```bash
tensorboard --logdir ai/logs/
```

Luego abre http://localhost:6006 en el navegador.

### Ajustar hiperparámetros

Edita `ai/agent.py`:
- `total_timesteps`: más iteraciones = mejor rendimiento (500k actual, probar 1M para mejores resultados)
- `learning_rate`: tasa de aprendizaje (default `3e-4`)
- `n_steps`: pasos por rollout (1024 actual, balance entre estabilidad y frecuencia de update)
- `gamma`: factor de descuento (0.995 para dar peso a recompensas futuras)
- `ent_coef`: coeficiente de exploración (0.01, evita que se estanque en policy subóptima)

Edita `ai/flappy_env.py`:
- `self._obstacle_interval`: frecuencia de obstáculos (0.6s actual)
- `gravity` / `jump_strength`: física del avión (450 / -450 actual)
- `Obstacle.speed`: velocidad de obstáculos (320 actual)
- Reward: `0.1` por frame, `+10` por obstáculo pasado, `-10` al morir

---

## Cómo ver a la IA jugar

```bash
python play.py
```

Se abrirá la ventana del juego con la IA jugando automáticamente.

---

## Estructura del proyecto

```
AgentesProject/
├── game/
│   ├── code/
│   │   ├── main.py        # Juego original (jugable por humano)
│   │   ├── settings.py    # WINDOW_WIDTH, WINDOW_HEIGHT, FRAMERATE
│   │   └── sprites.py     # BG, Ground, Plane, Obstacle
│   ├── graphics/          # Sprites e imágenes
│   └── sounds/            # Efectos de sonido
├── ai/
│   ├── flappy_env.py      # Entorno Gymnasium (OpenAI Gym)
│   ├── agent.py           # Wrapper de PPO (train, save, load, evaluate)
│   ├── train.py           # Script de entrenamiento
│   ├── evaluate.py        # (vacío — para evaluación futura)
│   └── checkpoints/       # Modelos guardados
└── main.py                # (vacío)
```
