import pyopencl as cl
import numpy as np
import gc


class GameOfLifeGPU:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        total_cells = grid_size * grid_size

        # === Initialisation OpenCL ===
        platforms = cl.get_platforms()
        gpu_devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
        self.context = cl.Context(devices=gpu_devices)
        self.queue = cl.CommandQueue(self.context)
        mem_flags = cl.mem_flags

        # === Buffers GPU ===
        self.grid_buffer = cl.Buffer(
            self.context, mem_flags.READ_WRITE, size=total_cells * np.dtype(np.uint8).itemsize
        )
        self.next_grid_buffer = cl.Buffer(
            self.context, mem_flags.READ_WRITE, size=self.grid_buffer.size
        )

        print("taille en mémoire :", self.grid_buffer.size*2 / 10**6, "Mo")

        # === Compilation du kernel OpenCL ===
        kernel_code = """
        // === Randomisation initiale ===
        uint pcg32(uint state) {
            uint oldstate = state * 747796405u + 2891336453u;
            uint xorshifted = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
            xorshifted ^= xorshifted >> 22u;
            return xorshifted;
        }

        __kernel void randomize_grid(__global uchar* grid, const uint seed) {
            const uint gid = get_global_id(0);
            uint state = gid + seed * 6364136223846793005u;
            uint rnd = pcg32(state);
            grid[gid] = (uchar)(rnd & 1u);
        }

        // === Étape de simulation ===
        __kernel void update(
            __global const uchar* current_grid,
            __global uchar* next_grid,
            const int grid_size
        ) {
            const int x = get_global_id(0);
            const int y = get_global_id(1);
            const size_t index = (size_t)y * grid_size + x;

            int neighbors = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = (x + dx + grid_size) % grid_size;
                    int ny = (y + dy + grid_size) % grid_size;
                    neighbors += (int) current_grid[(size_t)ny * grid_size + nx];
                }
            }

            uchar alive = (current_grid[index] && (neighbors == 2 || neighbors == 3)) ||
                          (!current_grid[index] && neighbors == 3);
            next_grid[index] = alive;
        }

        // === Extraction pour affichage ===
        __kernel void render_region(
            __global const uchar* grid,
            __global uchar* output_image,
            const int grid_size,
            const int x_offset,
            const int y_offset,
            const int width
        ) {
            const int x = get_global_id(0);
            const int y = get_global_id(1);
            const int gx = x + x_offset;
            const int gy = y + y_offset;

            if (gx >= grid_size || gy >= grid_size) return;

            const size_t grid_index = (size_t)gy * grid_size + gx;
            const size_t local_index = (size_t)y * width + x;

            output_image[local_index] = grid[grid_index];
        }
        """
        self.program = cl.Program(self.context, kernel_code).build()

        # === Randomisation sur GPU ===
        seed = np.int32(np.random.randint(0, 10**6))
        self.program.randomize_grid(
            self.queue,
            (total_cells,),
            None,
            self.grid_buffer,
            seed
        ).wait()

        gc.collect()

    # === Étape de simulation ===
    def step(self):
        self.program.update(
            self.queue,
            (self.grid_size, self.grid_size),
            None,
            self.grid_buffer,
            self.next_grid_buffer,
            np.int32(self.grid_size)
        ).wait()

        # Échange des buffers
        self.grid_buffer, self.next_grid_buffer = self.next_grid_buffer, self.grid_buffer

    # === Rendu d'une portion de la grille ===
    def render(self, x_start=0, y_start=0, x_end=None, y_end=None, rgba=True):
        if x_end is None:
            x_end = self.grid_size
        if y_end is None:
            y_end = self.grid_size

        width = x_end - x_start
        height = y_end - y_start
        total_pixels = width * height

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
            np.int32(x_start),
            np.int32(y_start),
            np.int32(width)
        ).wait()

        region_cpu = np.empty(total_pixels, dtype=np.uint8)
        cl.enqueue_copy(self.queue, region_cpu, tmp_buffer).wait()
        region_cpu = region_cpu.reshape((height, width))

        if rgba:
            image_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            image_rgba[..., 1] = region_cpu * 255  # Vert = vivant
            image_rgba[..., 3] = 255
            return image_rgba
        else:
            return region_cpu

    # === Libération des ressources ===
    def stop(self):
        del self.grid_buffer
        del self.next_grid_buffer
        del self.queue
        del self.context
        gc.collect()
