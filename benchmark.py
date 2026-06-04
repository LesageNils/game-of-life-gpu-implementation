import time
from PIL import Image

from gpu_life_separated_packed import GameOfLifeGPU as GameInt8

def benchmark(engine, name, benchmark_duration = 900, grid_size = 200_000, picture =False):
    """Exécute la simulation et mesure les FPS réels (CPU + GPU sync)."""
    print(f"Lancement du benchmark {name} sur une grille de {grid_size}x{grid_size}...")
    life_gpu = engine(grid_size)


    # Pré-chauffe : compilation JIT éventuelle + alloc
    life_gpu.step()
    life_gpu.queue.finish()

    print("Début de la mesure...")
    start = time.perf_counter()
    end_time = start + benchmark_duration
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
    print(f"Cellules/sec    : {(frames * grid_size ** 2) / elapsed / 1e12:.3f} Téra-cellules/s")
    print(f"Bande passante  : {((frames * (grid_size ** 2 / 8) * 2) / elapsed) / 1024 ** 3:.2f} Go/s (RW)")

    if picture:
        print("Rendu d'une frame de contrôle...")
        image = life_gpu.render(x_end=1920, y_end=1080)
        filename = f"gameoflife_{name.lower()}.png"
        Image.fromarray(image, mode="RGBA").save(filename)

    # Nettoyage
    life_gpu.stop()
    del life_gpu
    import gc
    gc.collect()

    return fps


if __name__ == "__main__":
    # === Lancement du test ===
    try:
        fps_int8 = benchmark(GameInt8, "optimized_packed")
        print(f"\nRésultat final : {fps_int8:.2f} FPS")
    except Exception as e:
        print(f"Erreur : {e}")