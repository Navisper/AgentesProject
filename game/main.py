# game/main.py
import pygame, sys

pygame.init()
screen = pygame.display.set_mode((288, 512))  # tamaño del juego
clock = pygame.time.Clock()

# Estado inicial del pájaro, tuberías, etc.
bird_y = screen.get_height()//2
velocity = 0

running = True
while running:
    for event in pygame.event.get():  # Procesar eventos
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            velocity = -10  # Flap: saltar

    # Lógica de físicas (gravidad, movimiento tuberías, colisiones, puntaje...)
    velocity += 1         # gravedad
    bird_y += velocity

    screen.fill((135, 206, 235))  # Limpiar pantalla (color cielo)
    # Aquí se dibujaría el pájaro, tuberías, piso, etc.
    # e.g.: pygame.draw.circle(screen, (255,0,0), (50, bird_y), 15)

    pygame.display.flip()  # Actualizar pantalla
    clock.tick(30)         # 30 FPS
