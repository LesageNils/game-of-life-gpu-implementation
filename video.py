import cv2
from tqdm import tqdm
from gpu_life_separated_packed import GameOfLifeGPU  # ou colle la classe directement ici


def generate_game_of_life_video(
    grid_size=1440, frames=300, output_file="game_of_life_1440p.mp4", fps=30
):
    # Initialisation du moteur GPU
    gol = GameOfLifeGPU(grid_size)

    # Codec H.264 (avc1)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    # Création du writer vidéo
    out = cv2.VideoWriter(output_file, fourcc, fps, (grid_size, grid_size))

    print(f"🎥 Génération de {frames} frames en {grid_size}x{grid_size} avec codec avc1...")

    for _ in tqdm(range(frames)):
        # Calcul et rendu d’une frame
        gol.step()
        image = gol.render()

        # Conversion RGBA → BGR pour OpenCV
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        # Écriture dans la vidéo
        out.write(bgr_image)

    # Libération des ressources
    out.release()
    gol.stop()

    print(f"✅ Vidéo enregistrée sous : {output_file}")


if __name__ == "__main__":
    generate_game_of_life_video(
        grid_size=3000,
        frames=300,  # ≈ 30 secondes à 30 fps
        output_file="images/game_of_life_1440p.mp4",
        fps=60
    )
