import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join


def create_relevant_patients_dataframe(protein_file, labels_file, normalization_type, output_dir, min_locations=3):
    # Read data
    protein_data = pd.read_excel(protein_file, engine='openpyxl')
    labels_data = pd.read_excel(labels_file, engine='openpyxl')

    # Get unique patient IDs
    unique_patient_ids = labels_data['patient number'].unique()

    # Identify columns for this normalization batch
    batch_columns = [col for col in protein_data.columns if col.startswith(normalization_type)]
    lysis_columns = [col for col in batch_columns if col.endswith('L')]

    # Select columns and create new dataframe
    relevant_columns = [col for col in protein_data.columns[:4]]  # Identifier columns

    for patient in tqdm(unique_patient_ids, desc=f'Processing {normalization_type} Batch'):
        patient_cols = [col for col in lysis_columns if str(patient) in col]

        if len(patient_cols) >= min_locations:
            relevant_columns.extend(patient_cols)

    # Create subset dataframe
    subset_df = protein_data[relevant_columns].copy()

    # Remove all-zero rows
    subset_df = subset_df[~(subset_df.iloc[:, 4:] == 0).all(axis=1)]

    # Save to CSV
    output_filename = join(output_dir, f'relevant_patients_proteomics_table_{normalization_type}.csv')
    subset_df.to_csv(output_filename, index=False)

    return subset_df


if __name__ == '__main__':
    csv_dir = "CSVs"
    protein_file = join(csv_dir, '2projects combined-proteinGroups-genes.xlsx')
    labels_file = join(csv_dir, '2projects combined labels.xlsx')

    for norm_type in ['Intensity', 'iBAQ', 'LFQ']:
        df = create_relevant_patients_dataframe(protein_file=protein_file, labels_file=labels_file,
                                                normalization_type=norm_type,
                                                output_dir=join(csv_dir, r"relevant_dataframes_per_norm_type"))

        print(f"{norm_type} dataframe shape: {df.shape}")
