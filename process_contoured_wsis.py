import os
import json
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

TIFFS_DIR = r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\Thesis\2021-01-17\Tiffs"
PROCESSED_WSIS_DIR = r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\Thesis\processed_wsis"
OUTPUT_DIR = r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\Thesis\averaged_slices"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_processing_stats(stats_path):
    try:
        with open(stats_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def load_tissue_regions(tissue_regions_path):
    try:
        with Image.open(tissue_regions_path) as img:
            return np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"Error loading tissue_regions.png: {e}")
        return None


def filter_slices(slices, processing_stats):
    valid_slices = []
    for idx, slice_data in enumerate(slices):
        if idx in processing_stats.get("valid_slices", []):
            valid_slices.append(slice_data)
    return valid_slices


def process_wsi(wsi_id, tiffs_dir, processed_dir, output_dir):
    try:
        wsi_processed_dir = os.path.join(processed_dir, wsi_id, f"{wsi_id}_Wholeslide_Default_Extended")
        tissue_regions_path = os.path.join(wsi_processed_dir, "tissue_regions.png")
        stats_path = os.path.join(wsi_processed_dir, "processing_stats.json")

        if not os.path.exists(tissue_regions_path) or not os.path.exists(stats_path):
            return f"Failed to process {wsi_id}: Missing required files: {tissue_regions_path}, {stats_path}"

        processing_stats = load_processing_stats(stats_path)
        if not processing_stats:
            return f"Failed to process {wsi_id}: Unable to load processing stats."

        tissue_regions = load_tissue_regions(tissue_regions_path)
        if tissue_regions is None:
            return f"Failed to process {wsi_id}: Unable to load tissue regions."

        wsi_tiff_dir = os.path.join(tiffs_dir, wsi_id)
        if not os.path.exists(wsi_tiff_dir):
            return f"Failed to process {wsi_id}: Missing Tiff directory {wsi_tiff_dir}"

        slices = []
        for tile_name in os.listdir(wsi_tiff_dir):
            tile_path = os.path.join(wsi_tiff_dir, tile_name)
            try:
                with Image.open(tile_path) as img:
                    slices.append(np.array(img, dtype=np.float32))
            except Exception as e:
                print(f"Error loading image {tile_path}: {e}")

        if not slices:
            return f"Failed to process {wsi_id}: No valid slices processed."

        valid_slices = filter_slices(slices, processing_stats)
        if not valid_slices:
            return f"Failed to process {wsi_id}: No valid slices after filtering."

        # Averaging slices
        average_slice = np.mean(valid_slices, axis=0).astype(np.uint8)

        # Saving averaged slices
        os.makedirs(os.path.join(output_dir, wsi_id), exist_ok=True)
        for idx, slice_data in enumerate(valid_slices):
            slice_path = os.path.join(output_dir, wsi_id, f"{wsi_id}_slice_{idx + 1}.jpg")
            Image.fromarray(slice_data.astype(np.uint8)).save(slice_path, "JPEG")

        averaged_slice_path = os.path.join(output_dir, wsi_id, f"{wsi_id}_averaged_slice.jpg")
        Image.fromarray(average_slice).save(averaged_slice_path, "JPEG")

        return f"Successfully processed {wsi_id}"

    except Exception as e:
        return f"Failed to process {wsi_id}: {e}"


def process_all_wsis(tiffs_dir, processed_dir, output_dir, n_workers=None):
    wsi_ids = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    results = []

    with ProcessPoolExecutor(max_workers=n_workers or os.cpu_count()) as executor:
        futures = [executor.submit(process_wsi, wsi_id, tiffs_dir, processed_dir, output_dir) for wsi_id in wsi_ids]
        for future in tqdm(futures, desc="Processing WSIs"):
            results.append(future.result())

    return results


if __name__ == "__main__":
    results = process_all_wsis(TIFFS_DIR, PROCESSED_WSIS_DIR, OUTPUT_DIR)
    for result in results:
        print(result)
