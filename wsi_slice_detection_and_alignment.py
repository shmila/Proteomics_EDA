import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import openslide
from tqdm import tqdm


class SliceDetector:
    def __init__(self, min_tissue_size: int = 1000):
        """
        Initialize slice detector with configurable parameters

        Args:
            min_tissue_size: Minimum area (in pixels) to consider as tissue
        """
        self.min_tissue_size = min_tissue_size
        self.chunk_buffer = None
        self.binary_buffer = None

    def _ensure_buffers(self, chunk_size: int):
        """Create or resize processing buffers if needed"""
        if (self.chunk_buffer is None or
                self.chunk_buffer.shape[0] != chunk_size):
            self.chunk_buffer = np.empty((chunk_size, chunk_size, 3),
                                         dtype=np.uint8)
            self.binary_buffer = np.empty((chunk_size, chunk_size),
                                          dtype=np.uint8)

    def process_chunk(self,
                      slide: openslide.OpenSlide,
                      x_start: int,
                      y_start: int,
                      width: int,
                      height: int) -> List[np.ndarray]:
        """
        Process a single chunk of the WSI

        Args:
            slide: OpenSlide object
            x_start, y_start: Starting coordinates
            width, height: Chunk dimensions

        Returns:
            List of contours found in chunk
            :param slide:
            :param height:
            :param width:
            :param y_start:
            :param x_start:
        """
        # Ensure buffers are right size
        self._ensure_buffers(max(width, height))

        # Read region into pre-allocated buffer
        chunk = np.array(slide.read_region(
            (x_start, y_start),
            0,
            (width, height)
        ).convert('RGB'))

        # Convert to grayscale (reuse buffer)
        gray = cv2.cvtColor(chunk, cv2.COLOR_RGB2GRAY)

        # Simple thresholding
        self.binary_buffer[:height, :width] = gray < 240
        binary = self.binary_buffer[:height, :width]

        # Find contours
        contours = cv2.findContours(
            binary.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )[0]

        # Filter and adjust contours
        valid_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > self.min_tissue_size:
                # Adjust coordinates to full image space
                contour += [x_start, y_start]
                valid_contours.append(contour)

        return valid_contours


def optimize_chunk_size(dimensions: Tuple[int, int],
                        target_chunks: int = 100) -> int:
    """Calculate optimal chunk size based on image dimensions"""
    width, height = dimensions
    area_per_chunk = (width * height) / target_chunks
    chunk_size = int(np.sqrt(area_per_chunk))
    return max(min(chunk_size, 10000), 1000)  # Between 1000 and 10000


def process_wsi_file(args: Tuple) -> Dict:
    """
    Process a single WSI file

    Args:
        args: Tuple of (wsi_path, output_dir, debug)

    Returns:
        Processing statistics dictionary
    """
    wsi_path, output_dir, debug = args
    start_time = datetime.now()
    filename = Path(wsi_path).name

    print(f"\nStarting processing of {filename}")
    print(f"Output directory will be: {output_dir}")

    try:
        # Create output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Open slide
        slide = openslide.OpenSlide(str(wsi_path))
        width, height = slide.dimensions

        if debug:
            print(f"\nProcessing {Path(wsi_path).name}")
            print(f"Dimensions: {width}x{height}")

        # Calculate chunk size
        chunk_size = optimize_chunk_size((width, height))
        n_chunks_w = (width + chunk_size - 1) // chunk_size
        n_chunks_h = (height + chunk_size - 1) // chunk_size

        if debug:
            print(f"Using chunk size: {chunk_size}")
            print(f"Processing {n_chunks_w}x{n_chunks_h} chunks")

        # Initialize detector
        detector = SliceDetector()

        print(f"Starting chunk processing...")

        # Process chunks
        tissue_regions = []
        chunk_pbar = tqdm(total=n_chunks_w * n_chunks_h,
                          desc=f"Processing chunks for {filename}")

        for y in range(n_chunks_h):
            for x in range(n_chunks_w):
                # Calculate chunk dimensions
                x_start = x * chunk_size
                y_start = y * chunk_size
                chunk_w = min(chunk_size, width - x_start)
                chunk_h = min(chunk_size, height - y_start)

                try:
                    # Process chunk
                    contours = detector.process_chunk(
                        slide, x_start, y_start, chunk_w, chunk_h
                    )
                    if contours:
                        print(f"Found {len(contours)} tissue regions in chunk {x},{y}")
                    tissue_regions.extend(contours)
                except Exception as e:
                    print(f"Warning: Error in chunk {x},{y} of {filename}: {str(e)}")

                chunk_pbar.update(1)

        chunk_pbar.close()
        print(f"Finished chunk processing. Found {len(tissue_regions)} total tissue regions")

        # Merge overlapping regions
        if tissue_regions:
            print("Starting region merging...")
            # Create small scale mask for merging
            scale = min(5000 / max(width, height), 1.0)
            mask_w = int(width * scale)
            mask_h = int(height * scale)
            print(f"Using scale factor {scale:.3f} for merging (output size: {mask_w}x{mask_h})")
            mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

            # Draw scaled contours
            scaled_contours = [(c * scale).astype(np.int32)
                               for c in tissue_regions]
            cv2.drawContours(mask, scaled_contours, -1, 255, -1)

            # Find contours in merged mask
            print("Finding contours in merged mask...")
            merged_contours = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )[0]

            # Scale back to original size
            print("Scaling contours back to original size...")
            merged_regions = [(c / scale).astype(np.int32)
                              for c in merged_contours]
            print(f"Merged into {len(merged_regions)} final regions")
        else:
            print("No tissue regions found to merge")
            merged_regions = []

        # Calculate statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        stats = {
            'file': str(wsi_path),
            'status': 'success',
            'dimensions': {'width': width, 'height': height},
            'processing': {
                'chunk_size': chunk_size,
                'n_chunks': n_chunks_w * n_chunks_h,
                'time_seconds': processing_time
            },
            'results': {
                'n_tissue_regions': len(merged_regions),
                'tissue_areas': [int(cv2.contourArea(c))
                                 for c in merged_regions]
            }
        }

        # Save results if output directory provided
        if output_dir:
            print(f"\nSaving results to {output_dir}")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save tissue region visualization
            if merged_regions:
                vis_path = output_dir / 'tissue_regions.png'
                print(f"Creating visualization image at {vis_path}")

                # Create visualization image
                vis_img = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
                scaled_contours = [(c * scale).astype(np.int32)
                                   for c in merged_regions]

                for contour in scaled_contours:
                    cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 2)

                cv2.imwrite(str(vis_path), vis_img)
                print(f"Saved visualization image")

            # Save stats
            stats_path = output_dir / 'processing_stats.json'
            print(f"Saving statistics to {stats_path}")
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print("Saved statistics file")

        return stats

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        return {
            'file': str(wsi_path),
            'status': 'failed',
            'error': str(e),
            'processing_time': processing_time
        }


def process_wsi_directory(base_dir: str,
                          output_base_dir: str,
                          n_workers: int = -1,
                          debug: bool = False) -> Dict:
    """
    Process all WSIs in a directory structure

    Args:
        base_dir: Base directory containing WSI files
        output_base_dir: Base directory for outputs
        n_workers: Number of worker processes (-1 for CPU count)
        debug: Whether to show debug output

    Returns:
        Processing statistics dictionary
    """
    try:
        base_dir = Path(base_dir)
        output_base_dir = Path(output_base_dir)

        print("Starting WSI processing...")
        print(f"Input directory: {base_dir}")
        print(f"Output directory: {output_base_dir}")

        # Get all tiff files
        tiff_files = []
        for wsi_dir in base_dir.iterdir():
            if wsi_dir.is_dir():
                tiffs = list(wsi_dir.glob("*.tif"))
                tiff_files.extend(tiffs)

        print(f"Found {len(tiff_files)} tiff files")

        # Setup parallel processing
        if n_workers <= 0:
            n_workers = mp.cpu_count()

        print(f"\nProcessing using {n_workers} workers...")

        # Prepare arguments
        process_args = [(str(tiff),
                         str(output_base_dir / tiff.parent.name / tiff.stem),
                         debug)
                        for tiff in tiff_files]

        # Process files in parallel
        results = []
        total_start_time = datetime.now()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(process_wsi_file, args): args[0]
                for args in process_args
            }

            # Process as they complete
            for future in tqdm(future_to_path,
                               total=len(tiff_files),
                               desc="Overall WSI Processing Progress"):
                wsi_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Print completion status
                    elapsed = (datetime.now() - total_start_time).total_seconds()
                    completed = len(results)
                    remaining = len(tiff_files) - completed
                    avg_time = elapsed / completed if completed > 0 else 0
                    est_remaining = avg_time * remaining if avg_time > 0 else 0

                    print(f"\nCompleted {Path(wsi_path).name}")
                    print(f"Progress: {completed}/{len(tiff_files)} files")
                    print(f"Average time per file: {avg_time:.1f}s")
                    print(f"Estimated remaining time: {est_remaining / 60:.1f} minutes")
                except Exception as e:
                    print(f"\nError processing {Path(wsi_path).name}: {str(e)}")

        # Compile statistics
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']

        # Calculate timing statistics
        processing_times = [r['processing']['time_seconds']
                            for r in successful]

        overall_stats = {
            'total_files': len(tiff_files),
            'successful': len(successful),
            'failed': len(failed),
            'timing': {
                'total_time': sum(processing_times),
                'mean_time': np.mean(processing_times) if processing_times else 0,
                'min_time': min(processing_times) if processing_times else 0,
                'max_time': max(processing_times) if processing_times else 0
            }
        }

        # Save overall statistics
        stats_file = output_base_dir / 'overall_processing_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(overall_stats, f, indent=2)

        # Print summary
        print(f"\nProcessing complete!")
        print(f"Total files: {len(tiff_files)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"\nTiming:")
        print(f"  Total time: {overall_stats['timing']['total_time']:.1f}s")
        print(f"  Mean time: {overall_stats['timing']['mean_time']:.1f}s")

        if failed:
            print("\nFailed files:")
            for result in failed:
                print(f"  {Path(result['file']).name}")
                print(f"    Error: {result['error']}")

        return overall_stats

    except Exception as e:
        print(f"Error in directory processing: {str(e)}")
        raise

    # Compile statistics
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    # Calculate timing statistics
    processing_times = [r['processing']['time_seconds']
                        for r in successful]

    overall_stats = {
        'total_files': len(tiff_files),
        'successful': len(successful),
        'failed': len(failed),
        'timing': {
            'total_time': sum(processing_times),
            'mean_time': np.mean(processing_times),
            'min_time': min(processing_times),
            'max_time': max(processing_times)
        }
    }

    # Save overall statistics
    stats_file = output_base_dir / 'overall_processing_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(overall_stats, f, indent=2)

    # Print summary
    print(f"\nProcessing complete!")
    print(f"Total files: {len(tiff_files)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"\nTiming:")
    print(f"  Total time: {overall_stats['timing']['total_time']:.1f}s")
    print(f"  Mean time: {overall_stats['timing']['mean_time']:.1f}s")

    if failed:
        print("\nFailed files:")
        for result in failed:
            print(f"  {Path(result['file']).name}")
            print(f"    Error: {result['error']}")

    return overall_stats


if __name__ == "__main__":
    # Process WSIs
    stats = process_wsi_directory(
        base_dir=r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\Thesis\2021-01-17\Tiffs",
        output_base_dir=r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\Thesis\processed_wsis",
        n_workers=mp.cpu_count(),  # Use all CPU cores
        debug=True
    )
