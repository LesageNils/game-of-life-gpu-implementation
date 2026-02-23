// Générateur aléatoire PCG32 (Permuted Congruential Generator)

uint pcg32_random(uint state) {
    uint oldstate = state;
    // Avancement de l'état (constantes LCG standard)
    state = oldstate * 747796405u + 2891336453u;
    // Sortie XSH-RR (XorShift High - Random Rotate)
    uint xorshifted = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
    xorshifted ^= xorshifted >> 22u;
    return xorshifted;
}

// INITIALISATION (Randomize)
__kernel void randomize_grid(
    __global uint* grid,
    const uint seed,
    const int width_in_uints
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width_in_uints) return;

    uint state = (uint)x;
    state ^= (uint)y * 2654435761u; // un grand nombre premier proche de 2^32 * phi
    state ^= seed * 0x9E3779B9u;    // Mélange avec la seed Python

    uint rand_val = pcg32_random(state);
    rand_val = pcg32_random(rand_val);

    ulong idx = (ulong)y * (ulong)width_in_uints + x;

    grid[idx] = rand_val;
}

// SIMULATION (Update)

__kernel void update(
    __global const uint* current_grid,
    __global uint* next_grid,
    const int grid_size,
    const int width_in_uints // nombre de colonnes de uint (grid_size / 32)
) {
    // x : BLOC de 32 cellules horizontales
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width_in_uints || y >= grid_size) return;

    // monde qui boucle
    int y_up   = (y == 0) ? grid_size - 1 : y - 1;
    int y_down = (y == grid_size - 1) ? 0 : y + 1;

    int x_left  = (x == 0) ? width_in_uints - 1 : x - 1;
    int x_right = (x == width_in_uints - 1) ? 0 : x + 1;

    ulong w = (ulong)width_in_uints;

    // Ligne du haut
    uint u_l = current_grid[(ulong)y_up * w + x_left];
    uint u_c = current_grid[(ulong)y_up * w + x];
    uint u_r = current_grid[(ulong)y_up * w + x_right];

    // Ligne du milieu (actuelle)
    uint m_l = current_grid[(ulong)y * w + x_left];
    uint m_c = current_grid[(ulong)y * w + x];
    uint m_r = current_grid[(ulong)y * w + x_right];

    // Ligne du bas
    uint d_l = current_grid[(ulong)y_down * w + x_left];
    uint d_c = current_grid[(ulong)y_down * w + x];
    uint d_r = current_grid[(ulong)y_down * w + x_right];

    // Voisins Haut
    uint n0 = (u_c << 1) | (u_l >> 31); // Nord-Ouest
    uint n1 = u_c;                      // Nord
    uint n2 = (u_c >> 1) | (u_r << 31); // Nord-Est

    // Voisins Milieu
    uint n3 = (m_c << 1) | (m_l >> 31); // Ouest
    uint n4 = (m_c >> 1) | (m_r << 31); // Est

    // Voisins Bas
    uint n5 = (d_c << 1) | (d_l >> 31); // Sud-Ouest
    uint n6 = d_c;                      // Sud
    uint n7 = (d_c >> 1) | (d_r << 31); // Sud-Est

    uint a = 0, b = 0, c = 0;

    // Optimisation : Groupement des additions
    // On ajoute n0..n7 à notre accumulateur (a,b,c)
    uint neighbors[] = {n0, n1, n2, n3, n4, n5, n6, n7};

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint n = neighbors[i];

        // Full Adder Logic: a + b + c + n
        uint t_a = a ^ n;
        uint carry_a = a & n;

        uint t_b = b ^ carry_a;
        uint carry_b = b & carry_a;

        c = c ^ carry_b;
        b = t_b;
        a = t_a;
    }

    // Vivant si : (Vivant ET 2 voisins) OU (3 voisins)
    // 2 voisins : c=0, b=1, a=0
    // 3 voisins : c=0, b=1, a=1

    uint two_neighbors = (~c) & b & (~a);

    uint three_neighbors = (~c) & b & a;


    uint result = (three_neighbors) | (two_neighbors & m_c);


    next_grid[(ulong)y * w + x] = result;
}


// RENDU (Unpacking pour affichage)
==
__kernel void render_region(
    __global const uint* grid,
    __global uchar* output_image,
    const int grid_size,
    const int width_in_uints,
    const int x_offset,
    const int y_offset,
    const int width
) {
    const int x = get_global_id(0); // Pixel x local
    const int y = get_global_id(1); // Pixel y local

    const int gx = x + x_offset;
    const int gy = y + y_offset;

    if (gx >= grid_size || gy >= grid_size) return;

    // Identification du mot et du bit
    int uint_x = gx / 32;
    int bit_pos = gx % 32;

    // Lecture sécurisée 64 bits
    ulong w = (ulong)width_in_uints;
    ulong idx = (ulong)gy * w + uint_x;

    uint word = grid[idx];

    uchar val = (word >> bit_pos) & 1;

    output_image[y * width + x] = val * 255;
}