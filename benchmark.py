import time
from PIL import Image

from gpu_life_separated_packed import GameOfLifeGPU as GameInt8

# Taille de la grille
GRID_SIZE = 250_000
BENCHMARK_DURATION = 15


def benchmark(life_gpu, name):
    """Exécute la simulation et mesure les FPS réels (CPU + GPU sync)."""
    print(f"Lancement du benchmark {name} sur une grille de {GRID_SIZE}x{GRID_SIZE}...")

    # Pré-chauffe : compilation JIT éventuelle + alloc
    life_gpu.step()
    life_gpu.queue.finish()

    print("Début de la mesure...")
    start = time.perf_counter()
    end_time = start + BENCHMARK_DURATION
    frames = 0

    while time.perf_counter() < end_time:
        life_gpu.step()
        frames += 1


    life_gpu.queue.finish()

    final_time = time.perf_counter()

    elapsed = final_time - start
    fps = frames / elapsed

    print(f"\n== Benchmark {name} ==")
    print(f"Frames simulées : {frames}")
    print(f"Durée réelle    : {elapsed:.3f} s")
    print(f"FPS moyen       : {fps:.2f}")
    print(f"Cellules/sec    : {(frames * GRID_SIZE ** 2) / elapsed / 1e12:.3f} Téra-cellules/s")
    print(f"Bande passante  : {((frames * (GRID_SIZE ** 2 / 8) * 2) / elapsed) / 1024 ** 3:.2f} Go/s (RW)")

    # Optionnel : Rendu
    print("Rendu d'une frame de contrôle...")
    image = life_gpu.render(x_end=1920, y_end=1080)
    filename = f"gameoflife_{name.lower()}.png"
    #Image.fromarray(image, mode="RGBA").save(filename)

    # Nettoyage
    life_gpu.stop()
    del life_gpu
    import gc
    gc.collect()

    return fps


if __name__ == "__main__":
    # === Lancement du test ===
    try:
        life_int8 = GameInt8(GRID_SIZE)
        fps_int8 = benchmark(life_int8, "optimized_packed")
        print(f"\nRésultat final : {fps_int8:.2f} FPS")
    except Exception as e:
        print(f"Erreur : {e}")