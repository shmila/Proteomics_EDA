import os
from pathlib import Path
import random
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit


class TilesDatasetCreator:
    def __init__(self, tiffs_base_dir, output_base_dir, slides_info_path,
                 num_rows=100, num_cols=100, test_size=0.2, val_size=0.2, seed=42):
        """
        Initialize the dataset creator

        Args:
            tiffs_base_dir: Base directory containing WSI data
            output_base_dir: Where to save the frozen tiles dataset
            slides_info_path: Path to SlidesZohar.xlsx
            num_rows: Number of rows in tile grid
            num_cols: Number of columns in tile grid
            test_size: Fraction of patients to use for testing
            val_size: Fraction of remaining patients (after test split) to use for validation
            seed: Random seed for reproducibility
        """
        self.tiffs_base_dir = Path(tiffs_base_dir)
        self.output_base_dir = Path(output_base_dir)
        self.slides_df = pd.read_excel(slides_info_path)
        self.test_size = test_size
        self.val_size = val_size
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.grid_size = f"{num_rows}x{num_cols}"
        random.seed(seed)

        print(f"\nInitializing TilesDatasetCreator:")
        print(f"Grid size: {self.grid_size}")
        print(f"Base directory: {self.tiffs_base_dir}")
        print(f"Output directory: {self.output_base_dir}")

        # Create directory structure with grid size in name
        self.tiles_dir = self.output_base_dir / f"tiles_dataset_{self.grid_size}"
        self.train_dir = self.tiles_dir / "train"
        self.val_dir = self.tiles_dir / "validation"
        self.test_dir = self.tiles_dir / "test"
        self.log_dir = self.output_base_dir / f"logs_{self.grid_size}"

        # Create all directories
        for directory in [self.train_dir, self.val_dir, self.test_dir, self.log_dir]:
            directory.mkdir(exist_ok=True, parents=True)
            print(f"Created directory: {directory}")

    def _find_matching_tile_directory(self, wsi_dir):
        """Find tile directory matching the specified grid size"""
        target_pattern = f"*tiles_{self.grid_size} good tiles"
        matching_dirs = list(wsi_dir.glob(target_pattern))

        if not matching_dirs:
            print(f"Warning: No {self.grid_size} tile directory found in {wsi_dir}")
            return None

        if len(matching_dirs) > 1:
            print(f"Warning: Multiple {self.grid_size} tile directories found in {wsi_dir}")
            print(f"Using first directory: {matching_dirs[0]}")

        print(f"Found matching tile directory: {matching_dirs[0]}")
        return matching_dirs[0]

    def _select_tiles(self, tile_dir, n_tiles=100):
        """Select tiles from a specific directory"""
        if tile_dir is None:
            return []

        tiles = list(tile_dir.glob("*.jpg"))
        print(f"Found {len(tiles)} tiles in {tile_dir}")

        if not tiles:
            return []

        if len(tiles) <= n_tiles:
            print(f"Warning: Only {len(tiles)} tiles available, using all")
            return tiles

        selected = random.sample(tiles, n_tiles)
        print(f"Selected {len(selected)} tiles randomly")
        return selected

    def _create_dataset_splits(self, dataset_info_df):
        """Create train/val/test splits patient-aware"""
        print("\nCreating dataset splits:")
        print(f"Total slides before split: {len(dataset_info_df)}")

        # First split out test set
        test_splitter = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=42)
        train_val_idx, test_idx = next(test_splitter.split(
            dataset_info_df, groups=dataset_info_df['patient_id']
        ))

        test_df = dataset_info_df.iloc[test_idx].copy()
        train_val_df = dataset_info_df.iloc[train_val_idx].copy()

        # Then split remaining data into train and validation
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=self.val_size, random_state=42)
        train_idx, val_idx = next(val_splitter.split(
            train_val_df, groups=train_val_df['patient_id']
        ))

        train_df = train_val_df.iloc[train_idx].copy()
        val_df = train_val_df.iloc[val_idx].copy()

        # Assign splits
        train_df['split'] = 'train'
        val_df['split'] = 'validation'
        test_df['split'] = 'test'

        print("\nSplit statistics:")
        print(f"Training set: {len(train_df)} slides from {len(train_df['patient_id'].unique())} patients")
        print(f"Validation set: {len(val_df)} slides from {len(val_df['patient_id'].unique())} patients")
        print(f"Test set: {len(test_df)} slides from {len(test_df['patient_id'].unique())} patients")

        return pd.concat([train_df, val_df, test_df])

    def _copy_tiles_to_split(self, tile_path, slide_id, idx, split):
        """Copy a tile to its appropriate split directory"""
        if split == 'train':
            output_dir = self.train_dir / slide_id
        elif split == 'validation':
            output_dir = self.val_dir / slide_id
        else:  # test
            output_dir = self.test_dir / slide_id

        output_dir.mkdir(exist_ok=True)
        new_name = f"{slide_id}_tile_{idx:03d}.jpg"
        shutil.copy2(tile_path, output_dir / new_name)
        return new_name

    def create_dataset(self, n_tiles_per_slide=100):
        """Create the frozen tiles dataset with train/val/test splits"""
        print(f"\nStarting dataset creation with {self.grid_size} grid size")
        print(f"Target tiles per slide: {n_tiles_per_slide}")

        dataset_info = []
        skipped_slides = []

        # First pass: collect all dataset information
        for _, row in tqdm(self.slides_df.iterrows(), total=len(self.slides_df),
                           desc=f"Collecting {self.grid_size} dataset information"):
            slide_id = row['SlideFile#']
            patient_id = row['Patient#']

            print(f"\nProcessing slide {slide_id} (Patient {patient_id}):")

            wsi_dir = self.tiffs_base_dir / slide_id
            if not wsi_dir.exists():
                print(f"Warning: Directory not found for slide {slide_id}")
                skipped_slides.append((slide_id, "Directory not found"))
                continue

            tile_dir = self._find_matching_tile_directory(wsi_dir)
            if not tile_dir:
                skipped_slides.append((slide_id, f"No {self.grid_size} tile directory"))
                continue

            selected_tiles = self._select_tiles(tile_dir, n_tiles_per_slide)
            if not selected_tiles:
                skipped_slides.append((slide_id, "No tiles selected"))
                continue

            dataset_info.append({
                'slide_id': slide_id,
                'patient_id': patient_id,
                'n_tiles': len(selected_tiles),
                'tumor_type': row['Tumor'],
                'subtype': row['Suptype'],
                'selected_tiles': selected_tiles
            })

        # Create DataFrame and perform splits
        print(f"\nCreating dataset splits for {len(dataset_info)} slides")
        dataset_df = pd.DataFrame(dataset_info)
        dataset_df = self._create_dataset_splits(dataset_df)

        # Second pass: Copy tiles to appropriate split directories
        print("\nCopying tiles to split directories:")
        for _, row in tqdm(dataset_df.iterrows(), total=len(dataset_df),
                           desc="Copying tiles"):
            for idx, tile_path in enumerate(row['selected_tiles']):
                self._copy_tiles_to_split(tile_path, row['slide_id'], idx, row['split'])

        # Remove the temporary selected_tiles column and save dataset information
        dataset_df = dataset_df.drop('selected_tiles', axis=1)
        dataset_df.to_csv(self.tiles_dir / "dataset_info.csv", index=False)

        # Create validation report
        self._create_validation_report(dataset_df, skipped_slides)

        return dataset_df

    def _create_validation_report(self, dataset_df, skipped_slides):
        """Create a detailed validation report"""
        report = []

        report.append(f"Dataset Creation Report - Grid Size: {self.grid_size}")
        report.append("=" * 50)

        # Overall statistics
        n_slides = len(dataset_df)
        n_complete = len(dataset_df[dataset_df['n_tiles'] == 100])

        report.append("\nOverall Statistics:")
        report.append("-" * 20)
        report.append(f"Total slides processed: {n_slides}")
        report.append(f"Slides with complete tile sets: {n_complete}")
        report.append(f"Slides with incomplete tile sets: {n_slides - n_complete}")
        report.append(f"Skipped slides: {len(skipped_slides)}")

        # Skipped slides details
        if skipped_slides:
            report.append("\nSkipped Slides Details:")
            report.append("-" * 20)
            for slide_id, reason in skipped_slides:
                report.append(f"Slide {slide_id}: {reason}")

        # Split statistics
        for split in ['train', 'validation', 'test']:
            split_df = dataset_df[dataset_df['split'] == split]
            report.append(f"\n{split.capitalize()} set:")
            report.append("-" * 20)
            report.append(f"  Slides: {len(split_df)}")
            report.append(f"  Patients: {len(split_df['patient_id'].unique())}")
            report.append(f"  Total tiles: {split_df['n_tiles'].sum()}")
            report.append("\n  Tumor types:")
            for tumor_type in split_df['tumor_type'].unique():
                if pd.isna(tumor_type):
                    continue
                n_tumor = len(split_df[split_df['tumor_type'] == tumor_type])
                report.append(f"    {tumor_type}: {n_tumor} slides")

        # Save report
        report_path = self.log_dir / f"dataset_creation_report_{self.grid_size}.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        print(f"\nValidation report saved to: {report_path}")


if __name__ == "__main__":
    # Paths
    TIFFS_BASE_DIR = r"C:\Users\eliran.shmi\Documents\Thesis\2021-01-17\Tiffs"
    OUTPUT_BASE_DIR = r"C:\Users\eliran.shmi\Documents\Thesis\dataset"
    SLIDES_INFO_PATH = r"C:\Users\eliran.shmi\Documents\Thesis\CSVs\SlidesZohar.xlsx"

    print(f"Starting dataset creation with grid size 100x100...")

    # Create dataset
    creator = TilesDatasetCreator(
        tiffs_base_dir=TIFFS_BASE_DIR,
        output_base_dir=OUTPUT_BASE_DIR,
        slides_info_path=SLIDES_INFO_PATH,
        num_rows=100,
        num_cols=100,
        test_size=0.2,
        val_size=0.2
    )

    dataset_df = creator.create_dataset(n_tiles_per_slide=100)
    print("\nDataset creation complete!")
    print(f"Final dataset size: {len(dataset_df)} slides")
    print(f"Output directory: {creator.tiles_dir}")