import os
import openslide
from PIL import Image
from tqdm import tqdm


def split_tiff_to_tiles(tiff_path, grid_size):
    # Open the slide using OpenSlide
    slide = openslide.OpenSlide(tiff_path)
    img_width, img_height = slide.dimensions

    # Define output folder (same as input folder)
    base_folder = os.path.dirname(tiff_path)
    base_name = os.path.basename(tiff_path).replace('.tiff', '')

    # Calculate tile size based on grid resolution
    rows, cols = grid_size
    tile_width = img_width // cols
    tile_height = img_height // rows

    # Create output folder if it doesn't exist
    output_folder = os.path.join(base_folder, f"{base_name}tiles{rows}x{cols}")
    os.makedirs(output_folder, exist_ok=True)

    total_tiles = rows * cols
    with tqdm(total=total_tiles, desc="Processing tiles", unit="tile") as pbar:
        # Extract and save each tile
        for i in range(rows):
            for j in range(cols):
                # Calculate the bounding box for the tile
                left = j * tile_width
                upper = i * tile_height
                right = left + tile_width
                lower = upper + tile_height

                # Read the region using OpenSlide
                tile = slide.read_region((left, upper), 0, (tile_width, tile_height))
                tile = tile.convert("RGB")

                # Construct the filename
                tile_filename = f"{base_name}tile{rows}x{cols}{i}{j}.png"
                tile_path = os.path.join(output_folder, tile_filename)

                # Save the tile
                tile.save(tile_path, format='PNG')

                # Update progress bar
                pbar.update(1)

    print(f"\nAll tiles saved in: {output_folder}")


# Example usage
tiff_file_path = "path/to/your/large_image.tiff"
split_tiff_to_tiles(tiff_file_path, grid_size=(64, 64))
