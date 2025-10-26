import time
import numpy as np
from PIL import Image

from gpu_life_separated import GameOfLifeGPU as GameInt8

# Taille de la grille
GRID_SIZE = 80000
BENCHMARK_DURATION = 15


def benchmark(life_gpu, name):
    """Exécute la simulation pendant un certain temps, puis sauvegarde la dernière image."""
    print(f"Lancement du benchmark {name} sur une grille de {GRID_SIZE}x{GRID_SIZE}...")

    # Pré-chauffe : un step initial
    life_gpu.step()

    start = time.perf_counter()
    end_time = start + BENCHMARK_DURATION
    frames = 0

    while time.perf_counter() < end_time:
        life_gpu.step()
        #life_gpu.render(x_end=1920, y_end=1080)
        frames += 1

    # Après la boucle, on rend la dernière image
    #image = life_gpu.render()

    elapsed = time.perf_counter() - start
    fps = frames / elapsed

    print(f"\n== Benchmark {name} ==")
    print(f"Frames simulées : {frames}")
    print(f"Durée mesurée   : {elapsed:.3f} s")
    print(f"FPS moyen       : {fps:.2f}")

    # Sauvegarde de la dernière image
    filename = f"gameoflife_{name.lower()}_{GRID_SIZE}px.png"
    #print(f"Sauvegarde de la dernière image dans {filename} ...")
    #Image.fromarray(image, mode="RGBA").save(filename)

    # Nettoyage GPU + RAM
    life_gpu.stop()
    del life_gpu
    import gc
    gc.collect()

    return fps


# === Lancement du test ===
life_int8 = GameInt8(GRID_SIZE)
fps_int8 = benchmark(life_int8, "int8")

print(f"\nRésultat final : {fps_int8:.2f} FPS")
