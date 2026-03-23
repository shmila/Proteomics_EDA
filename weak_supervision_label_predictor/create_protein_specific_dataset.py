import os
from pathlib import Path
import random
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
import json


class ProteinSpecificDatasetCreatorCV:
    def __init__(self, tiffs_base_dir, output_base_dir, slides_info_path, proteomics_path,
                 num_rows=100, num_cols=100, val_size=0.2, test_size=0.2, seed=42):
        """
        Initialize the dataset creator for a specific protein

        Args:
            tiffs_base_dir: Directory containing WSI tiff files and tile subdirs
            output_base_dir: Base directory for dataset output
            slides_info_path: Path to SlidesZohar.xlsx
            proteomics_path: Path to proteomics data Excel file
            num_rows/num_cols: Tile grid dimensions
            val_size/test_size: Split ratios
            seed: Random seed
        """
        self.tiffs_base_dir = Path(tiffs_base_dir)
        self.output_base_dir = Path(output_base_dir)
        self.slides_df = pd.read_excel(slides_info_path)
        self.proteomics_df = pd.read_excel(proteomics_path)
        self.val_size = val_size
        self.test_size = test_size
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.grid_size = f"{num_rows}x{num_cols}"
        random.seed(seed)
        np.random.seed(seed)

        print("\nInitializing ProteinSpecificDatasetCreator:")
        print(f"Grid size: {self.grid_size}")
        print(f"Base directory: {self.tiffs_base_dir}")
        print(f"Output directory: {self.output_base_dir}")

    def _get_patient_measurements(self, protein_row, norm_type):
        """Extract measurements for each patient from proteomics data"""
        prefix_map = {
            'Intensity': 'Intensity ',
            'iBAQ': 'iBAQ ',
            'LFQ': 'LFQ intensity '
        }
        prefix = prefix_map[norm_type]

        # Get all columns ending with 'L' (Lysis method)
        measurement_cols = [col for col in protein_row.index
                            if col.startswith(prefix) and col.endswith('L')]

        patient_measurements = {}
        for col in measurement_cols:
            patient = col.replace(prefix, '').split('_')[0]
            value = protein_row[col]

            if not pd.isna(value) and value > 0:
                if patient not in patient_measurements:
                    patient_measurements[patient] = []
                patient_measurements[patient].append(value)

        return patient_measurements

    def _calculate_protein_labels(self, protein_name, norm_type):
        """Calculate labels for all slides based on protein expression"""
        print(f"\nCalculating labels for protein: {protein_name}")
        print(f"Normalization type: {norm_type}")

        # Get protein row
        protein_mask = self.proteomics_df['Protein names'] == protein_name
        if not protein_mask.any():
            raise ValueError(f"Protein {protein_name} not found in proteomics data")

        protein_row = self.proteomics_df[protein_mask].iloc[0]

        # Get patient measurements
        patient_measurements = self._get_patient_measurements(protein_row, norm_type)

        # Calculate patient means
        patient_means = {}
        for patient, measurements in patient_measurements.items():
            if measurements:  # Only include patients with measurements
                patient_means[patient] = np.mean(measurements)

        if not patient_means:
            raise ValueError(f"No valid measurements found for protein {protein_name}")

        # Calculate median of means
        median_of_means = np.median(list(patient_means.values()))

        # Create labels for each slide
        labeled_slides = []
        for _, row in self.slides_df.iterrows():
            patient_id = str(row['Patient#'])
            slide_id = row['SlideFile#']

            if patient_id in patient_means:
                patient_mean = patient_means[patient_id]
                label = 1 if patient_mean > median_of_means else 0

                labeled_slides.append({
                    'slide_id': slide_id,
                    'patient_id': patient_id,
                    'label': label,
                    'expression_value': patient_mean,
                    'n_measurements': len(patient_measurements[patient_id])
                })

        labeled_df = pd.DataFrame(labeled_slides)

        # Print label distribution
        n_pos = sum(labeled_df['label'] == 1)
        n_neg = sum(labeled_df['label'] == 0)
        print("\nLabel distribution:")
        print(f"Positive samples: {n_pos} ({n_pos / len(labeled_df) * 100:.1f}%)")
        print(f"Negative samples: {n_neg} ({n_neg / len(labeled_df) * 100:.1f}%)")

        return labeled_df, median_of_means

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

        return matching_dirs[0]

    def _select_tiles(self, tile_dir, n_tiles=100):
        """Select tiles from a specific directory"""
        if tile_dir is None:
            return []

        tiles = list(tile_dir.glob("*.jpg"))
        print(f"Found {len(tiles)} tiles in {tile_dir}")

        if not tiles:
            return []

        # If n_tiles is None or we have fewer tiles than requested, return all tiles
        if n_tiles is None or len(tiles) <= n_tiles:
            print(f"Using all {len(tiles)} available tiles")
            return tiles

        selected = random.sample(tiles, n_tiles)
        print(f"Selected {len(selected)} tiles randomly")
        return selected

    def create_cv_folds(self, protein_name, norm_type='Intensity', n_folds=5):
        """Create cross-validation folds with patient-level splitting"""
        # Calculate labels
        labeled_df, threshold = self._calculate_protein_labels(protein_name, norm_type)

        # Get unique patients and their labels
        patient_labels = labeled_df.groupby('patient_id')['label'].first()
        unique_patients = pd.DataFrame({
            'patient_id': patient_labels.index,
            'label': patient_labels.values
        })

        # Create stratified folds at patient level
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_assignments = []

        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(unique_patients, unique_patients['label'])):
            # Split train_val into train and validation
            train_val_patients = unique_patients.iloc[train_val_idx]
            test_patients = unique_patients.iloc[test_idx]

            # Further split train_val maintaining stratification
            train_idx, val_idx = train_test_split(
                np.arange(len(train_val_patients)),
                test_size=0.2,
                stratify=train_val_patients['label'],
                random_state=42
            )

            train_patients = train_val_patients.iloc[train_idx]
            val_patients = train_val_patients.iloc[val_idx]

            # Assign slides to splits
            fold_df = labeled_df.copy()
            fold_df['fold'] = -1
            fold_df.loc[fold_df['patient_id'].isin(train_patients['patient_id']), 'split'] = 'train'
            fold_df.loc[fold_df['patient_id'].isin(val_patients['patient_id']), 'split'] = 'validation'
            fold_df.loc[fold_df['patient_id'].isin(test_patients['patient_id']), 'split'] = 'test'
            fold_df['fold'] = fold_idx

            fold_assignments.append(fold_df)

        # Combine all folds
        all_folds_df = pd.concat(fold_assignments)

        return all_folds_df, threshold

    def create_cv_datasets(self, protein_name, norm_type='Intensity', n_folds=5):
        """Create datasets for each fold"""
        all_folds_df, threshold = self.create_cv_folds(protein_name, norm_type, n_folds)

        # Create base dataset directory
        dataset_name = f"tiles_dataset_{self.grid_size}_{protein_name}_{norm_type}_cv"
        dataset_base_dir = self.output_base_dir / dataset_name

        print(f"\nCreating {n_folds}-fold CV dataset for {protein_name}")
        print(f"Using {norm_type} normalization")
        print(f"Output directory: {dataset_base_dir}")

        for fold in range(n_folds):
            print(f"\nProcessing fold {fold + 1}/{n_folds}")
            fold_dir = dataset_base_dir / f"fold_{fold}"
            fold_df = all_folds_df[all_folds_df['fold'] == fold]

            # Initialize lists to store all processed slides for this fold
            all_processed_slides = []
            all_skipped_slides = []

            # Process each split
            for split in ['train', 'validation', 'test']:
                split_df = fold_df[fold_df['split'] == split]
                split_dir = fold_dir / split
                split_dir.mkdir(parents=True, exist_ok=True)

                print(f"\nProcessing {split} set for fold {fold + 1}:")
                print(f"Found {len(split_df)} slides from {len(split_df['patient_id'].unique())} patients")

                # Process slides with progress bar
                for _, row in tqdm(split_df.iterrows(),
                                   total=len(split_df),
                                   desc=f"Processing {split} slides"):
                    slide_id = row['slide_id']
                    wsi_dir = self.tiffs_base_dir / slide_id
                    tile_dir = self._find_matching_tile_directory(wsi_dir)

                    if not tile_dir:
                        all_skipped_slides.append((slide_id, f"No {self.grid_size} tile directory"))
                        continue

                    # For test set, use all tiles; for train/val use sample
                    n_tiles = None if split == 'test' else 100
                    selected_tiles = self._select_tiles(tile_dir, n_tiles)

                    if not selected_tiles:
                        all_skipped_slides.append((slide_id, "No tiles selected"))
                        continue

                    # Create output directory and copy tiles
                    output_dir = split_dir / slide_id
                    output_dir.mkdir(exist_ok=True)

                    copied_tiles = []
                    for idx, tile_path in enumerate(selected_tiles):
                        new_name = f"{slide_id}_tile_{idx:03d}.jpg"
                        shutil.copy2(tile_path, output_dir / new_name)
                        copied_tiles.append(new_name)

                    # Add processed slide to the list
                    all_processed_slides.append({
                        **row.to_dict(),
                        'n_tiles': len(copied_tiles)
                    })

                # Print split statistics
                split_processed = [s for s in all_processed_slides if s['split'] == split]
                print(f"\n{split.capitalize()} set statistics for fold {fold + 1}:")
                print(f"Processed slides: {len(split_processed)}")
                if split_processed:
                    total_tiles = sum(slide['n_tiles'] for slide in split_processed)
                    print(f"Total tiles: {total_tiles}")
                    print(f"Average tiles per slide: {total_tiles / len(split_processed):.1f}")

            # Save fold info with ALL processed slides
            processed_df = pd.DataFrame(all_processed_slides)
            processed_df.to_csv(fold_dir / "dataset_info.csv", index=False)

            # Save fold statistics
            fold_stats = {
                'fold': fold,
                'n_slides': len(processed_df),
                'n_patients': len(processed_df['patient_id'].unique()),
                'skipped_slides': all_skipped_slides,
                'splits': {}
            }

            for split in ['train', 'validation', 'test']:
                split_data = processed_df[processed_df['split'] == split]
                pos_samples = split_data[split_data['label'] == 1]
                neg_samples = split_data[split_data['label'] == 0]

                fold_stats['splits'][split] = {
                    'n_slides': len(split_data),
                    'n_patients': len(split_data['patient_id'].unique()),
                    'n_tiles': int(split_data['n_tiles'].sum()),
                    'positive_samples': {
                        'slides': len(pos_samples),
                        'patients': len(pos_samples['patient_id'].unique())
                    },
                    'negative_samples': {
                        'slides': len(neg_samples),
                        'patients': len(neg_samples['patient_id'].unique())
                    }
                }

            with open(fold_dir / "fold_stats.json", 'w') as f:
                json.dump(fold_stats, f, indent=4)

        print(f"\nDataset creation complete!")
        print(f"Created {n_folds}-fold CV dataset at: {dataset_base_dir}")

        return dataset_base_dir


if __name__ == "__main__":
    # Paths
    THESIS_DIR = Path(r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\Thesis")
    TIFFS_DIR = THESIS_DIR / "2021-01-17" / "Tiffs"
    CSVS_DIR = THESIS_DIR / "CSVs"
    DATASET_DIR = THESIS_DIR / "dataset"

    SLIDES_INFO_PATH = CSVS_DIR / "SlidesZohar.xlsx"
    PROTEOMICS_PATH = CSVS_DIR / "2projects combined-proteinGroups-genes.xlsx"

    # Create dataset
    creator = ProteinSpecificDatasetCreatorCV(
        tiffs_base_dir=TIFFS_DIR,
        output_base_dir=DATASET_DIR,
        slides_info_path=SLIDES_INFO_PATH,
        proteomics_path=PROTEOMICS_PATH,
        num_rows=100,
        num_cols=100,
        val_size=0.2,
        seed=42
    )

    # Example protein
    # PROTEIN_NAME = " Thioredoxin"
    # PROTEIN_NAME = " Ubiquitin-conjugating enzyme E2 L3; Ubiquitin-conjugating enzyme E2 L5"

    # PROTEIN_NAMES = [" Elongation factor 1-alpha 1; Putative elongation factor 1-alpha-like 3"]
    # NORM_TYPES = ["Intensity", "LFQ"]

    # PROTEIN_NAMES = [" Ubiquitin-conjugating enzyme E2 L3; Ubiquitin-conjugating enzyme E2 L5"]
    # PROTEIN_NAMES = [" Gelsolin"]
    # PROTEIN_NAMES = [" 14-3-3 protein beta/alpha"]
    # PROTEIN_NAMES = [" Cornulin"]
    PROTEIN_NAMES = [" Proliferation marker protein Ki-67"]
    # NORM_TYPES = ["Intensity"]
    NORM_TYPES = ["LFQ"]

    N_FOLDS = 5

    for PROTEIN_NAME in PROTEIN_NAMES:
        for NORM_TYPE in NORM_TYPES:
            try:
                dataset_dir = creator.create_cv_datasets(
                    protein_name=PROTEIN_NAME,
                    norm_type=NORM_TYPE,
                    n_folds=N_FOLDS
                )
                print(f"\nCreated dataset for {PROTEIN_NAME} using {NORM_TYPE} normalization")
                print(f"Output directory: {dataset_dir}")
            except Exception as e:
                print(f"Error creating dataset: {str(e)}")
