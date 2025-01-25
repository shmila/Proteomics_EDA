import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from tqdm import tqdm


class ProteinLabeler:
    def __init__(self, dataset_base_dir, proteomics_data_path, norm_type='Intensity'):
        """
        Initialize the protein labeler

        Args:
            dataset_base_dir: Base directory containing the tiles dataset
            proteomics_data_path: Path to proteomics data Excel file
            norm_type: Type of normalization to use ('Intensity', 'iBAQ', or 'LFQ')
        """
        self.dataset_base_dir = Path(dataset_base_dir)
        self.proteomics_df = pd.read_excel(proteomics_data_path)
        self.norm_type = norm_type

        # Load dataset info with train/val/test splits
        self.dataset_info = pd.read_csv(self.dataset_base_dir / "dataset_info.csv")
        print(f"Loaded dataset info with {len(self.dataset_info)} slides")

        # Create directory for labels
        self.labels_dir = self.dataset_base_dir / f"protein_labels_{norm_type}"
        self.labels_dir.mkdir(exist_ok=True)

        # Verify we have all expected splits
        splits = set(self.dataset_info['split'].unique())
        expected_splits = {'train', 'validation', 'test'}
        if not splits.issuperset(expected_splits):
            missing = expected_splits - splits
            raise ValueError(f"Missing splits in dataset_info.csv: {missing}")

        print("Split distribution in dataset:")
        for split in expected_splits:
            n_slides = len(self.dataset_info[self.dataset_info['split'] == split])
            n_patients = len(self.dataset_info[self.dataset_info['split'] == split]['patient_id'].unique())
            print(f"  {split}: {n_slides} slides from {n_patients} patients")

    def _get_patient_measurements(self, protein_row):
        """Extract measurements for each patient from the proteomics data"""
        prefix_map = {
            'Intensity': 'Intensity ',
            'iBAQ': 'iBAQ ',
            'LFQ': 'LFQ intensity '
        }
        prefix = prefix_map[self.norm_type]

        # Get all columns for this normalization type that end with 'L' (Lysis method)
        measurement_cols = [col for col in protein_row.index
                            if col.startswith(prefix) and col.endswith('L')]

        patient_measurements = {}
        for col in measurement_cols:
            # Extract patient number from column name
            # Example: "Intensity 104_1L" -> "104"
            patient = col.replace(prefix, '').split('_')[0]

            value = protein_row[col]
            if not pd.isna(value) and value > 0:  # Only include valid measurements
                if patient not in patient_measurements:
                    patient_measurements[patient] = []
                patient_measurements[patient].append(value)

        return patient_measurements

    def _calculate_protein_statistics(self, protein_row):
        """Calculate patient-level statistics for a protein"""
        patient_measurements = self._get_patient_measurements(protein_row)
        patient_stats = {}

        # Calculate mean for each patient
        for patient, measurements in patient_measurements.items():
            if len(measurements) >= 1:  # Include patients with at least one measurement
                patient_stats[patient] = np.mean(measurements)

        if not patient_stats:
            return None

        # Calculate global mean (mean of patient means)
        global_mean = np.mean(list(patient_stats.values()))

        return {
            'patient_means': patient_stats,
            'global_mean': global_mean,
            'patient_measurements': patient_measurements
        }

    def create_protein_labels(self, protein_name):
        """Create labels for a specific protein"""
        print(f"\nCreating labels for protein: {protein_name}")
        print(f"Normalization type: {self.norm_type}")

        # Get protein row
        protein_mask = self.proteomics_df['Protein names'] == protein_name
        if not protein_mask.any():
            raise ValueError(f"Protein {protein_name} not found in proteomics data")

        protein_row = self.proteomics_df[protein_mask].iloc[0]

        # Calculate statistics
        stats = self._calculate_protein_statistics(protein_row)
        if stats is None:
            raise ValueError(f"No valid measurements found for protein {protein_name}")

        # Create labels for each slide
        labeled_slides = []
        for _, row in self.dataset_info.iterrows():
            patient_id = str(row['patient_id'])

            if patient_id in stats['patient_means']:
                patient_mean = stats['patient_means'][patient_id]
                label = 1 if patient_mean > stats['global_mean'] else 0

                labeled_slides.append({
                    'slide_id': row['slide_id'],
                    'patient_id': patient_id,
                    'split': row['split'],
                    'label': label,
                    'expression_value': patient_mean,
                    'n_measurements': len(stats['patient_measurements'][patient_id])
                })

        # Create DataFrame
        labels_df = pd.DataFrame(labeled_slides)

        # Create directory and save labels
        protein_dir = self.labels_dir / protein_name
        protein_dir.mkdir(exist_ok=True)

        labels_df.to_csv(protein_dir / "labels.csv", index=False)

        # Calculate and save statistics per split
        stats_dict = {
            'global_mean': stats['global_mean'],
            'normalization_type': self.norm_type
        }

        for split in ['train', 'validation', 'test']:
            split_df = labels_df[labels_df['split'] == split]
            stats_dict[split] = {
                'n_slides': len(split_df),
                'n_patients': len(split_df['patient_id'].unique()),
                'n_positive': sum(split_df['label'] == 1),
                'n_negative': sum(split_df['label'] == 0),
                'mean_expression': float(split_df['expression_value'].mean()),
                'std_expression': float(split_df['expression_value'].std()),
                'min_measurements': int(split_df['n_measurements'].min()),
                'max_measurements': int(split_df['n_measurements'].max()),
                'patient_list': split_df['patient_id'].unique().tolist()
            }

        with open(protein_dir / "statistics.json", 'w') as f:
            json.dump(stats_dict, f, indent=2)

        # Print summary
        print("\nLabel creation summary:")
        print(f"Global mean expression: {stats_dict['global_mean']:.2f}")
        for split in ['train', 'validation', 'test']:
            print(f"\n{split.capitalize()} set:")
            print(f"  Slides: {stats_dict[split]['n_slides']}")
            print(f"  Patients: {stats_dict[split]['n_patients']}")
            print(f"  Positive: {stats_dict[split]['n_positive']}")
            print(f"  Negative: {stats_dict[split]['n_negative']}")

        return labels_df


def create_labels_for_proteins(protein_list, dataset_base_dir, proteomics_data_path, norm_type='Intensity'):
    """Create labels for multiple proteins using specified normalization type"""
    # Initialize labeler
    labeler = ProteinLabeler(
        dataset_base_dir=dataset_base_dir,
        proteomics_data_path=proteomics_data_path,
        norm_type=norm_type
    )

    # Process each protein
    results = []
    for protein_name in tqdm(protein_list, desc="Creating protein labels"):
        try:
            labels_df = labeler.create_protein_labels(protein_name)
            results.append({
                'protein_name': protein_name,
                'status': 'success',
                'n_slides': len(labels_df),
                'norm_type': norm_type
            })
        except Exception as e:
            results.append({
                'protein_name': protein_name,
                'status': 'failed',
                'error': str(e),
                'norm_type': norm_type
            })

    # Save processing results
    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(dataset_base_dir) / f"protein_labeling_results_{norm_type}.csv",
                      index=False)

    return results_df


if __name__ == "__main__":
    # Paths
    DATASET_BASE_DIR = r"C:\Users\eliran.shmi\Documents\Thesis\dataset\tiles_dataset_100x100"
    PROTEOMICS_DATA_PATH = r"C:\Users\eliran.shmi\Documents\Thesis\CSVs\2projects combined-proteinGroups-genes.xlsx"

    # Example protein list
    proteins_of_interest = [
        " Thioredoxin",
        " DNA-directed RNA polymerases I and III subunit RPAC2",
        " UV excision repair protein RAD23 homolog B"
    ]

    # Specify normalization type
    NORM_TYPE = 'Intensity'  # or 'iBAQ' or 'LFQ'

    try:
        # Create labels for all proteins
        results = create_labels_for_proteins(
            proteins_of_interest,
            DATASET_BASE_DIR,
            PROTEOMICS_DATA_PATH,
            norm_type=NORM_TYPE
        )

        # Print summary
        print("\nLabeling Results Summary:")
        print(f"Normalization type: {NORM_TYPE}")
        print(f"Total proteins processed: {len(results)}")
        print(f"Successful: {sum(results['status'] == 'success')}")
        print(f"Failed: {sum(results['status'] == 'failed')}")

        # Print details of any failures
        failed = results[results['status'] == 'failed']
        if not failed.empty:
            print("\nFailed proteins:")
            for _, row in failed.iterrows():
                print(f"{row['protein_name']}: {row['error']}")

    except Exception as e:
        print(f"Error during label creation: {str(e)}")