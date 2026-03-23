import os
from pathlib import Path
import random
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
import json


class ProteinSpecificDatasetCreatorCV:
    def __init__(self, tiffs_base_dir, output_base_dir, slides_info_path, proteomics_path,
                 num_rows=100, num_cols=100, seed=42):
        """
        Initialize the dataset creator for CV

        Args:
            tiffs_base_dir: Directory containing WSI tiff files and tile subdirs
            output_base_dir: Base directory for dataset output
            slides_info_path: Path to SlidesZohar.xlsx
            proteomics_path: Path to proteomics data Excel file
            num_rows/num_cols: Tile grid dimensions
            seed: Random seed
        """
        self.tiffs_base_dir = Path(tiffs_base_dir)
        self.output_base_dir = Path(output_base_dir)
        self.slides_df = pd.read_excel(slides_info_path)
        self.proteomics_df = pd.read_excel(proteomics_path)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.grid_size = f"{num_rows}x{num_cols}"
        random.seed(seed)
        np.random.seed(seed)

        print("\nInitializing ProteinSpecificDatasetCreatorCV:")
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

        measurements = {}
        for col in protein_row.index:
            if col.startswith(prefix) and col.endswith('L'):
                patient = col.replace(prefix, '').split('_')[0]
                value = protein_row[col]
                if pd.notna(value) and value > 0:  # Only include valid measurements
                    if patient not in measurements:
                        measurements[patient] = []
                    measurements[patient].append(value)
        return measurements

    def _calculate_protein_labels(self, protein_name, norm_type):
        """Calculate labels for all slides based on protein expression"""
        print(f"\nCalculating labels for protein: {protein_name}")
        print(f"Normalization type: {norm_type}")

        # Get protein row
        protein_mask = self.proteomics_df['Protein names'] == protein_name
        if not protein_mask.any():
            raise ValueError(f"Protein {protein_name} not found")

        protein_row = self.proteomics_df[protein_mask].iloc[0]
        measurements = self._get_patient_measurements(protein_row, norm_type)

        # Calculate patient means
        patient_means = {
            patient: np.mean(vals)
            for patient, vals in measurements.items()
            if len(vals) >= 1  # Include patients with at least one measurement
        }

        if not patient_means:
            raise ValueError("No valid measurements found")

        # Calculate median of patient means for thresholding
        median_expression = np.median(list(patient_means.values()))

        # Create labels for each slide
        labeled_slides = []
        for _, row in self.slides_df.iterrows():
            patient_id = str(row['Patient#'])
            if patient_id in patient_means:
                labeled_slides.append({
                    'slide_id': row['SlideFile#'],
                    'patient_id': patient_id,
                    'label': 1 if patient_means[patient_id] > median_expression else 0,
                    'expression_value': patient_means[patient_id],
                    'n_measurements': len(measurements[patient_id])
                })

        df = pd.DataFrame(labeled_slides)
        print("\nLabel distribution:")
        print(f"Positive: {sum(df['label'] == 1)} ({100 * sum(df['label'] == 1) / len(df):.1f}%)")
        print(f"Negative: {sum(df['label'] == 0)} ({100 * sum(df['label'] == 0) / len(df):.1f}%)")
        return df, median_expression

    def create_cv_splits(self, labeled_df, n_folds=5):
        """Create cross-validation splits while maintaining label distribution"""
        import numpy as np
        from sklearn.model_selection import StratifiedGroupKFold

        print("\nCreating cross-validation splits...")

        # Create array for fold assignments
        result_df = labeled_df.copy()
        result_df['fold'] = -1
        result_df['split'] = ''

        # Get all patient IDs and their corresponding labels
        patient_data = result_df.groupby('patient_id')['label'].first().reset_index()

        # Create splits at patient level
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # First generate all splits to validate the distribution
        fold_assignments = []
        for fold_idx, (train_val_idx, test_idx) in enumerate(
                skf.split(patient_data, patient_data['label'], groups=patient_data['patient_id'])
        ):
            train_val_patients = patient_data.iloc[train_val_idx]['patient_id'].values
            test_patients = patient_data.iloc[test_idx]['patient_id'].values

            # Split train_val into train and validation
            train_val_df = patient_data.iloc[train_val_idx]
            val_patients = []

            # Get balanced validation set
            for label in [0, 1]:
                label_patients = train_val_df[train_val_df['label'] == label]['patient_id'].values
                n_val = max(1, int(len(label_patients) * 0.2))
                selected_val = np.random.choice(label_patients, size=n_val, replace=False)
                val_patients.extend(selected_val)

            # Remaining patients go to train
            train_patients = [p for p in train_val_patients if p not in val_patients]

            fold_assignments.append({
                'fold': fold_idx,
                'train': train_patients,
                'validation': val_patients,
                'test': test_patients
            })

        # Now apply the assignments to the result DataFrame
        for fold_info in fold_assignments:
            fold_idx = fold_info['fold']

            # Assign splits for this fold
            for split_name, patient_list in [
                ('train', fold_info['train']),
                ('validation', fold_info['validation']),
                ('test', fold_info['test'])
            ]:
                mask = result_df['patient_id'].isin(patient_list)
                result_df.loc[mask, 'split'] = split_name
                result_df.loc[mask, 'fold'] = fold_idx

            # Print statistics for this fold
            print(f"\nFold {fold_idx} statistics:")
            for split in ['train', 'validation', 'test']:
                split_df = result_df[(result_df['fold'] == fold_idx) &
                                     (result_df['split'] == split)]
                n_patients = len(split_df['patient_id'].unique())
                n_pos = sum(split_df['label'] == 1)
                n_neg = sum(split_df['label'] == 0)
                print(f"{split.capitalize()}:")
                print(f"  Patients: {n_patients}")
                print(f"  Slides: {len(split_df)} (Positive: {n_pos}, Negative: {n_neg})")

        # Verify all slides are assigned
        unassigned = result_df[result_df['split'] == '']
        if len(unassigned) > 0:
            print("\nWarning: Found unassigned slides:")
            print(unassigned)
            raise ValueError(f"Found {len(unassigned)} unassigned slides!")

        print("\nDebug: Fold distribution")
        fold_counts = result_df['fold'].value_counts().sort_index()
        print(fold_counts)

        print("\nDebug: Sample from each fold:")
        for fold_idx in range(n_folds):
            fold_sample = result_df[result_df['fold'] == fold_idx].head(1)
            if not fold_sample.empty:
                print(f"\nFold {fold_idx}:")
                print(fold_sample[['slide_id', 'patient_id', 'fold', 'split']])

        return result_df

    def _find_matching_tile_directory(self, wsi_dir):
        """Find tile directory matching the specified grid size"""
        pattern = f"*tiles_{self.grid_size} good tiles"
        matches = list(wsi_dir.glob(pattern))

        if not matches:
            print(f"Warning: No {self.grid_size} tile directory found in {wsi_dir}")
            return None

        if len(matches) > 1:
            print(f"Warning: Multiple {self.grid_size} directories found in {wsi_dir}")
            print(f"Using first directory: {matches[0]}")

        return matches[0]

    def _select_tiles(self, tile_dir, is_test_set=False, n_tiles=100):
        """
        Select tiles from a directory

        Args:
            tile_dir: Directory containing tiles
            is_test_set: Whether this is for test set (use all tiles if True)
            n_tiles: Number of tiles to select if not test set
        """
        if tile_dir is None or not tile_dir.exists():
            return []

        tiles = list(tile_dir.glob("*.jpg"))
        print(f"Found {len(tiles)} tiles in {tile_dir}")

        if not tiles:
            return []

        if is_test_set:
            print(f"Using all {len(tiles)} tiles for test set")
            return tiles

        n_select = min(len(tiles), n_tiles)
        if n_select < n_tiles:
            print(f"Warning: Only {n_select} tiles available")
        else:
            print(f"Selected {n_select} tiles randomly")

        return random.sample(tiles, n_select)

    def create_cv_datasets(self, protein_name, norm_type='Intensity', n_folds=5):
        """Create cross-validation datasets for a specific protein"""
        # Calculate labels and create splits
        split_df = self._calculate_protein_labels(protein_name, norm_type)[0]
        labeled_df = self.create_cv_splits(split_df, n_folds)

        # Create the output directory structure
        dataset_name = f"tiles_dataset_{self.grid_size}_{protein_name}_{norm_type}_cv"
        dataset_dir = self.output_base_dir / dataset_name

        print(f"\nCreating {n_folds}-fold CV dataset for {protein_name}")
        print(f"Using {norm_type} normalization")
        print(f"Output directory: {dataset_dir}")

        # Get all unique folds and make sure they're what we expect
        unique_folds = sorted(labeled_df['fold'].unique())
        if len(unique_folds) != n_folds:
            raise ValueError(f"Expected {n_folds} folds, but found {len(unique_folds)}")
        print(f"\nProcessing {len(unique_folds)} folds: {unique_folds}")

        # Process each fold
        for fold_idx in unique_folds:
            print(f"\nProcessing fold {fold_idx}")
            fold_dir = dataset_dir / f"fold_{fold_idx}"

            # Get data for this specific fold
            fold_data = labeled_df[labeled_df['fold'] == fold_idx].copy()
            if len(fold_data) == 0:
                print(f"Warning: No data found for fold {fold_idx}")
                continue

            print(f"Found {len(fold_data)} slides in fold {fold_idx}")

            # Track all processed slides for this fold
            all_processed_slides = []

            # Process each split
            for split in ['train', 'validation', 'test']:
                split_data = fold_data[fold_data['split'] == split].copy()
                split_dir = fold_dir / split
                split_dir.mkdir(parents=True, exist_ok=True)

                print(f"\nProcessing {split} set for fold {fold_idx + 1}:")
                print(f"Found {len(split_data)} slides from {len(split_data['patient_id'].unique())} patients")

                # Process each slide in this split
                for _, row in tqdm(split_data.iterrows(), desc=f"Processing {split} slides"):
                    slide_id = row['slide_id']
                    wsi_dir = self.tiffs_base_dir / slide_id
                    tile_dir = self._find_matching_tile_directory(wsi_dir)

                    if not tile_dir:
                        continue

                    # Select tiles - all for test, sample for train/val
                    is_test = split == 'test'
                    selected_tiles = self._select_tiles(tile_dir, is_test)

                    if not selected_tiles:
                        continue

                    # Create output directory for this slide
                    output_dir = split_dir / slide_id
                    output_dir.mkdir(exist_ok=True)

                    # Copy tiles
                    copied_tiles = []
                    for idx, tile_path in enumerate(selected_tiles):
                        new_name = f"{slide_id}_tile_{idx:03d}.jpg"
                        shutil.copy2(tile_path, output_dir / new_name)
                        copied_tiles.append(new_name)

                    # Store processed slide info
                    processed_slide = {
                        'slide_id': row['slide_id'],
                        'patient_id': row['patient_id'],
                        'label': row['label'],
                        'expression_value': row['expression_value'],
                        'n_measurements': row['n_measurements'],
                        'split': split,
                        'fold': fold_idx,
                        'n_tiles': len(copied_tiles)
                    }
                    all_processed_slides.append(processed_slide)

                # Print split statistics
                split_processed = [s for s in all_processed_slides if s['split'] == split]
                print(f"\n{split.capitalize()} set statistics for fold {fold_idx + 1}:")
                print(f"Processed slides: {len(split_processed)}")
                if split_processed:
                    total_tiles = sum(slide['n_tiles'] for slide in split_processed)
                    print(f"Total tiles: {total_tiles}")
                    print(f"Average tiles per slide: {total_tiles / len(split_processed):.1f}")

            # Save fold information
            if all_processed_slides:
                processed_df = pd.DataFrame(all_processed_slides)
                info_path = fold_dir / "dataset_info.csv"
                processed_df.to_csv(info_path, index=False)
                print(f"\nSaved dataset info for fold {fold_idx} to: {info_path}")
                print(f"Number of processed slides: {len(processed_df)}")

        print(f"\nDataset creation complete!")
        return dataset_dir


if __name__ == "__main__":
    # Paths
    THESIS_DIR = Path(r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\Thesis")
    TIFFS_DIR = THESIS_DIR / "2021-01-17" / "Tiffs"
    CSVS_DIR = THESIS_DIR / "CSVs"
    DATASET_DIR = THESIS_DIR / "dataset"

    SLIDES_INFO_PATH = CSVS_DIR / "SlidesZohar.xlsx"
    PROTEOMICS_PATH = CSVS_DIR / "2projects combined-proteinGroups-genes.xlsx"

    PROTEIN_NAMES = [" Thioredoxin"]
    NORM_TYPES = ["Intensity"]
    N_FOLDS = 5

    creator = ProteinSpecificDatasetCreatorCV(
        tiffs_base_dir=TIFFS_DIR,
        output_base_dir=DATASET_DIR,
        slides_info_path=SLIDES_INFO_PATH,
        proteomics_path=PROTEOMICS_PATH
    )

    for PROTEIN_NAME in PROTEIN_NAMES:
        for NORM_TYPE in NORM_TYPES:
            try:
                dataset_dir = creator.create_cv_datasets(  # Changed from create_cv_dataset
                    protein_name=PROTEIN_NAME,
                    norm_type=NORM_TYPE,
                    n_folds=N_FOLDS
                )
                print(f"\nCreated dataset for {PROTEIN_NAME} using {NORM_TYPE} normalization")
                print(f"Output directory: {dataset_dir}")
            except Exception as e:
                print(f"Error creating dataset: {str(e)}")
