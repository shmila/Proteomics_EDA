import os
from pathlib import Path
import numpy as np
import cv2
import openslide
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WSISliceProcessor:
    def __init__(self, min_slice_area: int = 10000,
                 similarity_threshold: float = 0.85,
                 alignment_params: Dict = None):
        """
        Initialize WSI slice processor

        Args:
            min_slice_area: Minimum area to consider as valid slice
            similarity_threshold: Threshold for considering slices similar
            alignment_params: Parameters for alignment algorithm
        """
        self.min_slice_area = min_slice_area
        self.similarity_threshold = similarity_threshold
        self.alignment_params = alignment_params or {
            'max_iterations': 50,
            'termination_eps': 1e-3,
            'motion_model': cv2.MOTION_EUCLIDEAN
        }

    def load_existing_results(self, wsi_id: str, processed_wsis_dir: Path) -> Tuple[List[np.ndarray], Dict]:
        """
        Load existing contour detection results

        Args:
            wsi_id: WSI identifier (e.g., "4M23")
            processed_wsis_dir: Base directory containing processed results
        """
        # Construct paths to results
        result_dir = processed_wsis_dir / wsi_id / f"{wsi_id}_Wholeslide_Default_Extended"
        regions_path = result_dir / 'tissue_regions.png'
        stats_path = result_dir / 'processing_stats.json'

        if not regions_path.exists() or not stats_path.exists():
            raise FileNotFoundError(
                f"Missing required files in {result_dir}. "
                f"Regions exists: {regions_path.exists()}, "
                f"Stats exists: {stats_path.exists()}"
            )

        # Load contour image
        tissue_img = cv2.imread(str(regions_path))
        if tissue_img is None:
            raise ValueError(f"Failed to read tissue regions image from {regions_path}")

        # Extract contours
        gray = cv2.cvtColor(tissue_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Load stats
        with open(stats_path) as f:
            stats = json.load(f)

        return contours, stats

    def group_similar_slices(self, contours: List[np.ndarray]) -> List[List[Dict]]:
        """Group similar contours based on shape and size"""
        valid_contours = []

        # First pass: Get basic metrics
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_slice_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            moments = cv2.moments(contour)
            centroid = (
                int(moments['m10'] / moments['m00']) if moments['m00'] != 0 else x,
                int(moments['m01'] / moments['m00']) if moments['m00'] != 0 else y
            )

            valid_contours.append({
                'contour': contour,
                'area': area,
                'bbox': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'centroid': centroid
            })

        # Second pass: Group similar contours
        groups = []
        used_indices = set()

        for i, slice1 in enumerate(valid_contours):
            if i in used_indices:
                continue

            current_group = [slice1]
            used_indices.add(i)

            for j, slice2 in enumerate(valid_contours):
                if j in used_indices:
                    continue

                if self._are_slices_similar(slice1, slice2):
                    current_group.append(slice2)
                    used_indices.add(j)

            groups.append(current_group)

        return groups

    def _are_slices_similar(self, slice1: Dict, slice2: Dict) -> bool:
        """Compare two slices for similarity using multiple metrics"""
        # Compare areas
        area_ratio = slice1['area'] / slice2['area']
        if not (0.7 <= area_ratio <= 1.3):
            return False

        # Compare bounding box dimensions
        w1 = slice1['bbox'][2]
        h1 = slice1['bbox'][3]
        w2 = slice2['bbox'][2]
        h2 = slice2['bbox'][3]

        width_ratio = w1 / w2
        height_ratio = h1 / h2

        if not (0.7 <= width_ratio <= 1.3 and 0.7 <= height_ratio <= 1.3):
            return False

        # Compare shapes with Hu moments
        shape_match = cv2.matchShapes(
            slice1['contour'],
            slice2['contour'],
            cv2.CONTOURS_MATCH_I1,
            0
        )

        # Compare centroids distance relative to size
        dx = slice1['centroid'][0] - slice2['centroid'][0]
        dy = slice1['centroid'][1] - slice2['centroid'][1]
        centroid_dist = np.sqrt(dx * dx + dy * dy)
        max_size = max(w1, h1, w2, h2)

        return (shape_match < self.similarity_threshold and
                centroid_dist < max_size * 0.5)

    def extract_and_align_slices(self, wsi_path: str,
                                 contour_groups: List[List[Dict]]) -> List[np.ndarray]:
        """Extract and align slices from each group"""
        # Open slide
        slide = openslide.OpenSlide(wsi_path)
        aligned_groups = []

        for group_idx, group in enumerate(contour_groups):
            try:
                # First pass: determine reference size and collect regions
                regions = []
                ref_width = ref_height = 0

                for slice_info in group:
                    x, y, w, h = slice_info['bbox']
                    region = np.array(slide.read_region((x, y), 0, (w, h)).convert('RGB'))
                    regions.append(region)
                    ref_width = max(ref_width, region.shape[1])
                    ref_height = max(ref_height, region.shape[0])

                if not regions:
                    continue

                # Second pass: normalize all regions to reference size
                normalized_regions = []
                for region in regions:
                    if region.shape[:2] != (ref_height, ref_width):
                        region = cv2.resize(region, (ref_width, ref_height))
                    if len(region.shape) != 3 or region.shape[2] != 3:
                        logger.warning(f"Skipping malformed region in group {group_idx}")
                        continue
                    normalized_regions.append(region)

                if not normalized_regions:
                    continue

                # Now all regions should be the same size
                try:
                    aligned_group = self._align_and_average_group(normalized_regions)
                    if aligned_group is not None and len(aligned_group.shape) == 3:
                        aligned_groups.append(aligned_group)
                except Exception as align_error:
                    logger.warning(f"Error aligning group {group_idx}: {str(align_error)}")
                    continue

            except Exception as e:
                logger.warning(f"Error processing group {group_idx}: {str(e)}")
                continue

        return aligned_groups

    def _align_and_average_group(self, regions: List[np.ndarray]) -> Optional[np.ndarray]:
        """Align and average a group of regions"""
        if not regions:
            return None

        # Ensure all regions have the same shape
        shapes = [region.shape for region in regions]
        if len(set(str(shape) for shape in shapes)) != 1:
            raise ValueError(f"All regions must have the same shape. Got shapes: {shapes}")

        # Convert to float32 for averaging
        regions_float = [region.astype(np.float32) for region in regions]

        # Average per channel
        result = np.zeros_like(regions_float[0])
        for channel in range(3):
            channel_stack = np.stack([region[..., channel] for region in regions_float])
            result[..., channel] = np.mean(channel_stack, axis=0)

        return result.astype(np.uint8)

        return result.astype(np.uint8)

    def _preprocess_for_alignment(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better alignment"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Normalize intensity
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(normalized, (5, 5), 0)

        # Enhance edges
        edges = cv2.Canny(blurred, 50, 150)

        return edges

    def _align_single_region(self, moving: np.ndarray,
                             reference: np.ndarray,
                             original_moving: np.ndarray) -> Optional[np.ndarray]:
        """Align a single region using enhanced ECC"""
        try:
            # Initialize transformation matrix
            warp_matrix = np.eye(2, 3, dtype=np.float32)

            # Try alignment with different motion models
            motion_models = [
                cv2.MOTION_TRANSLATION,
                cv2.MOTION_EUCLIDEAN,
                cv2.MOTION_AFFINE
            ]

            aligned = None
            for motion_model in motion_models:
                try:
                    criteria = (
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        self.alignment_params['max_iterations'],
                        self.alignment_params['termination_eps']
                    )

                    _, transform = cv2.findTransformECC(
                        reference, moving, warp_matrix,
                        motion_model, criteria
                    )

                    # Apply transformation to original image
                    aligned = cv2.warpAffine(
                        original_moving,
                        transform,
                        (original_moving.shape[1], original_moving.shape[0]),
                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                    )
                    break

                except cv2.error:
                    continue

            if aligned is None:
                # Fallback to simple centroid alignment
                aligned = self._align_by_centroid(
                    original_moving, reference
                )

            return aligned

        except Exception as e:
            logger.warning(f"Alignment failed: {str(e)}")
            return None

    def _align_by_centroid(self, moving: np.ndarray, reference: np.ndarray) -> Optional[np.ndarray]:
        """Simple alignment using centroids"""
        try:
            # Calculate centroids
            m = cv2.moments(moving)
            r = cv2.moments(reference)

            if m['m00'] == 0 or r['m00'] == 0:
                return None

            cx_m = int(m['m10'] / m['m00'])
            cy_m = int(m['m01'] / m['m00'])
            cx_r = int(r['m10'] / r['m00'])
            cy_r = int(r['m01'] / r['m00'])

            # Create translation matrix
            dx = cx_r - cx_m
            dy = cy_r - cy_m
            return np.float32([[1, 0, dx], [0, 1, dy]])

        except Exception as e:
            logger.warning(f"Centroid alignment failed: {str(e)}")
            return None

    def save_results(self, output_dir: Path,
                     aligned_groups: List[np.ndarray],
                     final_average: np.ndarray) -> None:
        """Save aligned slices and final average"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual aligned slices
        slices_dir = output_dir / 'aligned_slices'
        slices_dir.mkdir(exist_ok=True)

        for i, group in enumerate(aligned_groups):
            slice_path = slices_dir / f'slice_{i + 1}.jpg'
            cv2.imwrite(str(slice_path), cv2.cvtColor(group, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Save final averaged result
        final_path = output_dir / 'final_averaged_slice.jpg'
        cv2.imwrite(str(final_path),
                    cv2.cvtColor(final_average, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    def _find_alignment_transform(self, moving: np.ndarray, reference: np.ndarray) -> Optional[np.ndarray]:
        """Find transformation with improved error handling"""
        try:
            # Convert to grayscale
            moving_gray = cv2.cvtColor(moving, cv2.COLOR_RGB2GRAY)
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)

            # Normalize intensities
            moving_gray = cv2.normalize(moving_gray, None, 0, 255, cv2.NORM_MINMAX)
            ref_gray = cv2.normalize(ref_gray, None, 0, 255, cv2.NORM_MINMAX)

            # Apply Gaussian blur
            moving_gray = cv2.GaussianBlur(moving_gray, (5, 5), 0)
            ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)

            # Try different motion models in order of complexity
            motion_models = [
                cv2.MOTION_TRANSLATION,
                cv2.MOTION_EUCLIDEAN,
                cv2.MOTION_AFFINE
            ]

            for motion_model in motion_models:
                try:
                    warp_matrix = np.eye(2, 3, dtype=np.float32)
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3)

                    _, warp_matrix = cv2.findTransformECC(
                        ref_gray,
                        moving_gray,
                        warp_matrix,
                        motion_model,
                        criteria,
                        None,
                        5  # gaussFiltSize
                    )
                    return warp_matrix
                except cv2.error:
                    continue

            # If all motion models fail, fall back to simpler alignment
            return self._align_by_centroid(moving_gray, ref_gray)

        except Exception as e:
            logger.warning(f"Could not find transformation: {str(e)}")
            return None


def process_single_wsi(args: Tuple) -> Dict:
    """Process a single WSI file"""
    wsi_path, output_dir, processed_wsis_dir = args

    try:
        # Setup paths
        wsi_path = Path(wsi_path)
        output_dir = Path(output_dir)
        processed_wsis_dir = Path(processed_wsis_dir)

        # Extract WSI ID and name
        wsi_id = wsi_path.parent.name  # e.g., "4M23"
        wsi_name = f"{wsi_id}_Wholeslide_Default_Extended"
        output_path = output_dir / wsi_id

        logger.info(f"\nProcessing WSI {wsi_id}")
        logger.info(f"WSI path: {wsi_path}")
        logger.info(f"Results path: {processed_wsis_dir / wsi_id}")

        # Initialize processor
        processor = WSISliceProcessor()

        # Load existing results
        try:
            contours, stats = processor.load_existing_results(wsi_id, processed_wsis_dir)
        except FileNotFoundError as e:
            logger.error(f"Error loading results: {str(e)}")
            return {
                'wsi': wsi_id,
                'status': 'error',
                'message': str(e)
            }

        # Group similar slices
        slice_groups = processor.group_similar_slices(contours)
        logger.info(f"Found {len(slice_groups)} distinct slice groups")

        # Extract and align slices
        aligned_groups = processor.extract_and_align_slices(str(wsi_path), slice_groups)

        if not aligned_groups:
            return {
                'wsi': wsi_id,
                'status': 'error',
                'message': 'No valid aligned groups produced'
            }

        # Create final average
        final_average = np.mean(aligned_groups, axis=0).astype(np.uint8)

        # Save results
        processor.save_results(output_path, aligned_groups, final_average)

        return {
            'wsi': wsi_id,
            'status': 'success',
            'n_groups': len(aligned_groups),
            'output_dir': str(output_path)
        }

    except Exception as e:
        logger.error(f"Error processing {wsi_id}: {str(e)}")
        return {
            'wsi': wsi_id,
            'status': 'error',
            'message': str(e)
        }


def process_all_wsis(base_dir: str, output_base_dir: str, processed_wsis_dir: str, n_workers: int = -1) -> None:
    """
    Process all WSIs in parallel

    Args:
        base_dir: Base directory containing WSI TIFF files
        output_base_dir: Base directory for aligned slice outputs
        processed_wsis_dir: Directory containing contour detection results
        n_workers: Number of worker processes (-1 for CPU count)
    """
    try:
        base_dir = Path(base_dir)
        output_base_dir = Path(output_base_dir)
        processed_wsis_dir = Path(processed_wsis_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)

        # Verify processed_wsis_dir exists
        if not processed_wsis_dir.exists():
            raise ValueError(f"Processed WSIs directory not found: {processed_wsis_dir}")

        # Get all WSI files from the Tiffs directory
        wsi_files = []
        logger.info(f"Looking for WSIs in: {base_dir}")

        for wsi_dir in base_dir.iterdir():
            if wsi_dir.is_dir():
                # Look for the main WSI file
                tiff_file = wsi_dir / f"{wsi_dir.name}_Wholeslide_Default_Extended.tif"
                if tiff_file.exists():
                    logger.info(f"Found WSI: {tiff_file}")
                    wsi_files.append(tiff_file)
                else:
                    logger.warning(f"No TIFF found at expected path: {tiff_file}")

        logger.info(f"Found {len(wsi_files)} WSI files")

        # Setup parallel processing
        if n_workers <= 0:
            n_workers = mp.cpu_count()

        logger.info(f"Processing using {n_workers} workers")

        # Prepare arguments
        process_args = [
            (str(wsi), str(output_base_dir), str(processed_wsis_dir))
            for wsi in wsi_files
        ]

        # Process files in parallel
        results = []
        total_start_time = datetime.now()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Process as they complete with progress bar
            with tqdm(total=len(wsi_files), desc="Processing WSIs") as pbar:
                # Submit all tasks
                future_to_path = {
                    executor.submit(process_single_wsi, args): args[0]
                    for args in process_args
                }

                # Process as they complete
                for future in future_to_path:
                    wsi_path = future_to_path[future]
                    try:
                        result = future.result()
                        results.append(result)

                        # Update progress
                        pbar.update(1)

                        # Print completion status
                        elapsed = (datetime.now() - total_start_time).total_seconds()
                        completed = len(results)
                        remaining = len(wsi_files) - completed
                        avg_time = elapsed / completed if completed > 0 else 0
                        est_remaining = avg_time * remaining if avg_time > 0 else 0

                        logger.info(f"\nCompleted {Path(wsi_path).name}")
                        logger.info(f"Progress: {completed}/{len(wsi_files)} files")
                        logger.info(f"Average time per file: {avg_time:.1f}s")
                        logger.info(f"Estimated remaining time: {est_remaining / 60:.1f} minutes")

                    except Exception as e:
                        logger.error(f"Error processing {Path(wsi_path).name}: {str(e)}")

        # Calculate statistics
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']

        # Calculate timing statistics
        total_time = (datetime.now() - total_start_time).total_seconds()

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(wsi_files),
            'successful': len(successful),
            'failed': len(failed),
            'timing': {
                'total_time_seconds': total_time,
                'average_time_per_file': total_time / len(wsi_files) if wsi_files else 0
            },
            'failed_details': [{
                'wsi': r['wsi'],
                'error': r['message']
            } for r in failed]
        }

        # Save summary
        summary_path = output_base_dir / 'processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Print final summary
        logger.info("\nProcessing Summary:")
        logger.info(f"Total WSIs processed: {len(wsi_files)}")
        logger.info(f"Successfully processed: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total processing time: {total_time / 60:.1f} minutes")

        if failed:
            logger.info(f"\nFailed WSIs: {len(failed)}")
            # Print only first 5 failures as examples
            for result in failed[:5]:
                logger.info(f"  {result['wsi']}: {result['message']}")
            if len(failed) > 5:
                logger.info(f"  ... and {len(failed) - 5} more failures")

    except Exception as e:
        logger.error(f"Error in main processing loop: {str(e)}")
        raise


if __name__ == "__main__":
    # Base directories
    THESIS_DIR = Path(r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\Thesis")
    TIFFS_DIR = THESIS_DIR / "2021-01-17" / "Tiffs"
    PROCESSED_WSIS_DIR = THESIS_DIR / "processed_wsis"
    OUTPUT_DIR = THESIS_DIR / "processed_wsis_aligned"

    # Verify paths
    logger.info("Verifying paths:")
    logger.info(f"THESIS_DIR exists: {THESIS_DIR.exists()}")
    logger.info(f"TIFFS_DIR exists: {TIFFS_DIR.exists()}")
    logger.info(f"PROCESSED_WSIS_DIR exists: {PROCESSED_WSIS_DIR.exists()}")

    # Verify we have the expected structure in processed_wsis
    test_wsi = "4M23"
    test_result_path = PROCESSED_WSIS_DIR / test_wsi / f"{test_wsi}_Wholeslide_Default_Extended" / "tissue_regions.png"
    logger.info(f"Testing result path exists: {test_result_path.exists()}")

    try:
        # Process all WSIs
        process_all_wsis(
            base_dir=TIFFS_DIR,
            output_base_dir=OUTPUT_DIR,
            processed_wsis_dir=PROCESSED_WSIS_DIR,
            n_workers=mp.cpu_count()
        )
        logger.info("Processing completed successfully!")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
