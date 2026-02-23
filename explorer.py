import sys
import math
import time
import numpy as np
import pygame
from pygame.locals import *
from gpu_life_separated_packed import GameOfLifeGPU


# Configuration initiale

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
GRID_SIZE = 250000
START_CELL_SIZE = 4



def read_full_grid(gpu: GameOfLifeGPU):
    """Lit toute la grille GPU dans un numpy array uint8 (shape (N,N))."""
    cpu_buf = np.empty((gpu.grid_size, gpu.grid_size), dtype=np.uint8)
    cl = __import__("pyopencl")
    cl.enqueue_copy(gpu.command_queue, cpu_buf, gpu.grid_buffer)
    gpu.command_queue.finish()
    return cpu_buf


def write_full_grid(gpu: GameOfLifeGPU, cpu_grid):
    """Écrit toute la grille CPU dans le buffer GPU (remplace le contenu)."""
    cl = __import__("pyopencl")
    cl.enqueue_copy(gpu.command_queue, gpu.grid_buffer, cpu_grid)
    gpu.command_queue.finish()


def toggle_cell(gpu: GameOfLifeGPU, x, y):
    """Inverse la cellule (x,y) — lit-modifie-écrit la grille entière.
    (Simple, robuste. Pour optimisation on pourrait copier un seul octet.)
    """
    cpu = read_full_grid(gpu)
    if 0 <= x < gpu.grid_size and 0 <= y < gpu.grid_size:
        cpu[y, x] = 0 if cpu[y, x] else 1
        write_full_grid(gpu, cpu)


# -------------------------
# Main Pygame application
# -------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), RESIZABLE)
    pygame.display.set_caption("Game of Life — GPU explorer")

    clock = pygame.time.Clock()

    # Instanciation du backend GPU
    gpu = GameOfLifeGPU(GRID_SIZE)

    # Vue (en coordonnées de cellules)
    cell_size = START_CELL_SIZE  # pixels par cellule
    offset_x = 0
    offset_y = 0

    dragging = False
    drag_start_px = (0, 0)
    drag_start_offset = (0, 0)

    last_step_time = time.time()
    step_interval = 1.0  # secondes entre steps automatiques
    running = True
    paused = False
    speed = 1

    font = pygame.font.SysFont(None, 18)

    while running:
        now = time.time()
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            elif event.type == VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, RESIZABLE)

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_SPACE:
                    paused = not paused
                elif event.key == K_RIGHT:
                    # étape manuelle
                    gpu.step()
                elif event.key == K_f:
                    # étape manuelle
                    step_interval = step_interval / 2
                elif event.key == K_s:
                    # étape manuelle
                    step_interval = step_interval * 2

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 3:  # bouton droit -> start dragging
                    dragging = True
                    drag_start_px = event.pos
                    drag_start_offset = (offset_x, offset_y)
                elif event.button == 4:  # molette vers le haut -> zoom in
                    mx, my = event.pos
                    old_cell_size = cell_size
                    new_cell_size = min(128, max(1, cell_size + 1))
                    if new_cell_size != old_cell_size:
                        # conserver le point sous la souris
                        mouse_grid_x = offset_x + mx // old_cell_size
                        mouse_grid_y = offset_y + my // old_cell_size
                        cell_size = new_cell_size
                        offset_x = int(mouse_grid_x - mx // cell_size)
                        offset_y = int(mouse_grid_y - my // cell_size)
                elif event.button == 5:  # molette vers le bas -> zoom out
                    mx, my = event.pos
                    old_cell_size = cell_size
                    new_cell_size = min(128, max(1, cell_size - 1))
                    if new_cell_size != old_cell_size:
                        mouse_grid_x = offset_x + mx // old_cell_size
                        mouse_grid_y = offset_y + my // old_cell_size
                        cell_size = new_cell_size
                        offset_x = int(mouse_grid_x - mx // cell_size)
                        offset_y = int(mouse_grid_y - my // cell_size)

            elif event.type == MOUSEBUTTONUP:
                if event.button == 3:
                    dragging = False

            elif event.type == MOUSEMOTION:
                if dragging:
                    # pan: déplacer offset en cellules selon déplacement pixel
                    dx_px = event.pos[0] - drag_start_px[0]
                    dy_px = event.pos[1] - drag_start_px[1]
                    dx_cells = int(dx_px / cell_size)
                    dy_cells = int(dy_px / cell_size)
                    offset_x = drag_start_offset[0] - dx_cells
                    offset_y = drag_start_offset[1] - dy_cells

        # Clamp offset pour rester dans la grille
        view_w_px, view_h_px = screen.get_size()
        cells_w = max(1, math.ceil(view_w_px / cell_size))
        cells_h = max(1, math.ceil(view_h_px / cell_size))
        if offset_x < 0: offset_x = 0
        if offset_y < 0: offset_y = 0
        if offset_x > gpu.grid_size - cells_w:
            offset_x = max(0, gpu.grid_size - cells_w)
        if offset_y > gpu.grid_size - cells_h:
            offset_y = max(0, gpu.grid_size - cells_h)

        # Simulation: step toutes les step_interval secondes (si pas en pause)
        if not paused and (now - last_step_time) >= step_interval:
            start_time = time.time()
            gpu.step()
            end_time = time.time()
            last_step_time = now
            speed = - start_time + end_time

        # Rendu via GPU: on demande la portion visible
        x_start = int(offset_x)
        y_start = int(offset_y)
        x_end = min(gpu.grid_size, x_start + cells_w)
        y_end = min(gpu.grid_size, y_start + cells_h)

        # render renvoie un tableau (height, width, 4) dtype=uint8 en RGBA

        region = gpu.render(x_start, y_start, x_end, y_end)
        # region.shape -> (cells_h, cells_w, 4) normalement

        # Créer une surface pygame à partir du buffer RGBA
        h_cells = region.shape[0]
        w_cells = region.shape[1]
        # frombuffer attend (bytes) et dimension (width, height)
        surf = pygame.image.frombuffer(region.tobytes(), (w_cells, h_cells), "RGBA")
        # Échelle selon cell_size
        target_w = w_cells * cell_size
        target_h = h_cells * cell_size
        if cell_size != 1:
            surf = pygame.transform.scale(surf, (target_w, target_h))

        # Blit sur l'écran
        screen.fill((20, 20, 20))
        screen.blit(surf, (0, 0))

        # UI overlay (info)
        info_lines = [
            f"Grid: {gpu.grid_size}x{gpu.grid_size}",
            f"Cell size: {cell_size}px",
            f"Offset: ({offset_x}, {offset_y}) — Visible cells: {w_cells}×{h_cells}",
            f"Paused: {paused} (Space pour toggle) — Step every {step_interval:.1f}s",
            "Controls: clic gauche toggle cellule | clic droit + drag pan | molette zoom | ←/→ step"
        ]
        y = 6
        for line in info_lines:
            txt = font.render(line, True, (220, 220, 220))
            screen.blit(txt, (6, y))
            y += 18

        pygame.display.flip()
        pygame.display.set_caption(
            "Game of Life — GPU explorer - " + str(1 / step_interval) + " Hz - " + str(int(1 / speed)) + " step fps max")
        clock.tick(60)  # framerate plafonné, mais simulation indépendante

    # Clean up
    gpu.stop()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
