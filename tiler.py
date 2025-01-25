import os
from os.path import join

import openslide
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import cv2


def is_tissue_tile(tile_array, tissue_threshold=0.1, concentration_threshold=0.7):
    """
    Check if a tile contains sufficient tissue content considering spatial distribution.

    Args:
        tile_array: numpy array of RGB image (H&E stained)
        tissue_threshold: minimum fraction of non-background pixels required
        concentration_threshold: how concentrated the signal should be in the non-empty part
    """
    # Convert to grayscale
    gray = cv2.cvtColor(tile_array, cv2.COLOR_RGB2GRAY)

    # Background pixels will be very bright
    signal_mask = gray < 240

    # Calculate total signal fraction
    signal_fraction = np.sum(signal_mask) / signal_mask.size

    if signal_fraction < tissue_threshold:
        return False

    # Divide image into quadrants and check signal distribution
    h, w = signal_mask.shape
    quadrants = [
        signal_mask[:h // 2, :w // 2],  # top-left
        signal_mask[:h // 2, w // 2:],  # top-right
        signal_mask[h // 2:, :w // 2],  # bottom-left
        signal_mask[h // 2:, w // 2:]  # bottom-right
    ]

    # Calculate signal fraction in each quadrant
    quadrant_fractions = [np.sum(q) / q.size for q in quadrants]

    # If signal is concentrated (high fraction in some quadrants, low in others)
    # this suggests it's a border tile
    max_quadrant = max(quadrant_fractions)
    signal_concentration = max_quadrant / (signal_fraction + 1e-6)  # avoid div by 0

    return signal_concentration >= concentration_threshold


def split_wsi_to_tiles(tiff_path, num_rows=100, num_cols=100, tissue_threshold=0.9, max_workers=4):
    """
    Split a WSI into tiles with tissue content verification.
    """
    # Open the slide using OpenSlide
    slide = openslide.OpenSlide(tiff_path)
    img_width, img_height = slide.dimensions

    # Calculate tile dimensions based on desired grid
    tile_width = img_width // num_cols
    tile_height = img_height // num_rows

    # Setup output directory
    base_folder = os.path.dirname(tiff_path)
    base_name = os.path.basename(tiff_path).replace('.tif', '')
    output_folder = os.path.join(base_folder, f"{base_name}_tiles_{num_rows}x{num_cols} good tiles")
    os.makedirs(output_folder, exist_ok=True)

    def process_tile_openslide(args):
        """Process a single tile with tissue content verification"""
        i, j, output_path = args
        try:
            # Extract tile using OpenSlide
            left = j * tile_width
            upper = i * tile_height
            tile = slide.read_region((left, upper), 0, (tile_width, tile_height))
            tile = tile.convert('RGB')
            tile_array = np.array(tile)

            # Use debug mode for first few tiles
            debug_mode = (i == 0 and j < 5)  # Debug first 5 tiles of first row

            # Check tissue content
            if is_tissue_tile(tile_array, tissue_threshold):
                output_path = output_path.replace('.png', '.jpg')  # Change to jpg
                tile.save(output_path, format='JPEG', quality=90)  # Save as JPEG with quality=90
                if debug_mode:
                    tile.save(f'debug_tile_{i}_{j}.jpg', format='JPEG', quality=90)  # Save debug tiles as JPEG too
                return True
            return False

        except Exception as e:
            print(f"Error processing tile at ({i}, {j}): {str(e)}")
            return False

    # Prepare arguments for parallel processing
    tile_args = []
    for i in range(num_rows):
        for j in range(num_cols):
            output_path = os.path.join(output_folder, f"{base_name}_tile_{i}_{j}.png")
            tile_args.append((i, j, output_path))

    # Process tiles in parallel
    valid_tiles = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_tile_openslide, tile_args),
            total=len(tile_args),
            desc="Processing tiles"
        ))
        valid_tiles = sum(results)

    print(f"\nProcessing complete:")
    print(f"Total tiles processed: {len(tile_args)}")
    print(f"Valid tiles (≥{tissue_threshold * 100}% tissue): {valid_tiles}")
    print(f"Tiles saved in: {output_folder}")

    return output_folder, valid_tiles


# Example usage
if __name__ == "__main__":
    tiffs_based_dir = r"C:\Users\eliran.shmi\Documents\Thesis\2021-01-17\Tiffs"
    # tiff_file_relative_path = r"1M21\1M21_Wholeslide_Default_Extended.tif"

    for subdir in os.listdir(tiffs_based_dir):
        # if subdir.startswith("5") or subdir.startswith("3") or subdir == "2M01" or subdir == "2M02":
        if subdir == "1M25":
            for file in os.listdir(join(tiffs_based_dir, subdir)):
                if file.endswith("tif"):
                    tiff_file_path = join(tiffs_based_dir, subdir, file)
                    print(f"creating 100x100 tiles for {tiff_file_path}")
                    output_dir, n_valid_tiles = split_wsi_to_tiles(
                        tiff_file_path,
                        tissue_threshold=0.9
        )
