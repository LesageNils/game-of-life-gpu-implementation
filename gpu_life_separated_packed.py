import time

import pyopencl as cl
import numpy as np
import gc


class GameOfLifeGPU:
    def __init__(self, grid_size):
        # Pour une efficacité maximale (32 bits), la grille doit être multiple de 32
        if grid_size % 32 != 0:
            grid_size = (grid_size // 32 + 1) * 32
            print(f"Grille ajustée à {grid_size}x{grid_size} pour alignement 32-bits.")

        self.grid_size = grid_size

        # Dimensions en "mots" de 32 bits (colonnes compressées)
        self.width_in_uints = self.grid_size // 32
        total_bits = self.grid_size * self.grid_size

        # Taille en octets : (N*N) / 8
        buffer_size = total_bits // 8

        # === Initialisation OpenCL ===
        platforms = cl.get_platforms()
        print(f"Plateformes trouvées : {platforms}")

        best_device = None
        max_memory = 0

        # 1. On parcourt toutes les plateformes disponibles
        for platform in platforms:
            # On cherche d'abord les GPU dédiés/disponibles sur cette plateforme
            devices = platform.get_devices(device_type=cl.device_type.GPU)

            # Si aucun GPU n'est trouvé, on se rabat sur tous les types (CPU, Integrated, etc.)
            if not devices:
                devices = platform.get_devices(device_type=cl.device_type.ALL)

            # 2. On compare la mémoire de chaque appareil trouvé
            for device in devices:
                # global_mem_size renvoie la taille en octets
                if device.global_mem_size > max_memory:
                    max_memory = device.global_mem_size
                    best_device = device

        # 3. Validation et affichage du résultat
        if best_device:
            print("--- En fonction de la mémoire globale ---")
            print(f"Appareil sélectionné : {best_device.name.strip()}")
            print(f"Plateforme associée  : {best_device.platform.name.strip()}")
            print(f"Mémoire globale      : {max_memory / (1024 ** 3):.2f} Go")
        else:
            print("Aucun appareil OpenCL n'a été trouvé.")

        self.context = cl.Context(devices=[best_device])
        self.queue = cl.CommandQueue(self.context)
        mem_flags = cl.mem_flags

        # === Buffers GPU (Bit-packed) ===
        # On utilise des uint32 (4 octets) pour le traitement
        self.grid_buffer = cl.Buffer(
            self.context, mem_flags.READ_WRITE, size=buffer_size
        )
        self.next_grid_buffer = cl.Buffer(
            self.context, mem_flags.READ_WRITE, size=buffer_size
        )

        print(f"Taille en mémoire : {buffer_size / 1024 ** 2:.2f} Mo (Optimisé x8)")

        # Lecture du kernel (supposé être dans kernel_packed.c)
        with open("kernel_packed.c", "r") as f:
            kernel_code = f.read()

        self.program = cl.Program(self.context, kernel_code).build()

        # === Randomisation ===
        seed = np.uint32(np.random.randint(0, 0xFFFFFFFF, dtype=np.uint64))

        global_work_size = (self.width_in_uints, self.grid_size)

        self.program.randomize_grid(
            self.queue,
            global_work_size,
            None,
            self.grid_buffer,
            seed,
            np.int32(self.width_in_uints)
        ).wait()

        gc.collect()

    def step(self):
        # On lance le kernel sur une grille 2D où X est le nombre de mots de 32 bits
        global_work_size = (self.width_in_uints, self.grid_size)

        self.program.update(
            self.queue,
            global_work_size,
            None,
            self.grid_buffer,
            self.next_grid_buffer,
            np.int32(self.grid_size),
            np.int32(self.width_in_uints)
        ).wait()

        # Échange des buffers
        self.grid_buffer, self.next_grid_buffer = self.next_grid_buffer, self.grid_buffer

    def render(self, x_start=0, y_start=0, x_end=None, y_end=None, rgba=True):
        if x_end is None: x_end = self.grid_size
        if y_end is None: y_end = self.grid_size

        width = x_end - x_start
        height = y_end - y_start
        total_pixels = width * height

        # Buffer temporaire pour récupérer des octets (0 ou 255) pour l'affichage
        tmp_buffer = cl.Buffer(
            self.context,
            cl.mem_flags.WRITE_ONLY,
            total_pixels * np.dtype(np.uint8).itemsize
        )

        self.program.render_region(
            self.queue,
            (width, height),
            None,
            self.grid_buffer,
            tmp_buffer,
            np.int32(self.grid_size),
            np.int32(self.width_in_uints),
            np.int32(x_start),
            np.int32(y_start),
            np.int32(width)
        ).wait()

        region_cpu = np.empty(total_pixels, dtype=np.uint8)
        cl.enqueue_copy(self.queue, region_cpu, tmp_buffer).wait()
        region_cpu = region_cpu.reshape((height, width))

        if rgba:
            image_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            # Vert fluo pour le style
            image_rgba[..., 1] = region_cpu  # Canal Vert
            image_rgba[..., 3] = 255  # Alpha
            return image_rgba
        else:
            return region_cpu

    def stop(self):
        del self.grid_buffer
        del self.next_grid_buffer
        del self.queue
        del self.context
        gc.collect()