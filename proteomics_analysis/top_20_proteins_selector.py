from os.path import join

import pandas as pd
import numpy as np
from tqdm import tqdm


def create_protein_analysis_dataframe(csv_file, norm_type):
    df = pd.read_csv(csv_file)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Analyzing Proteins'):
        identifiers = row.iloc[:4]

        # Find measurement columns
        if norm_type == 'LFQ':
            lysis_columns = [col for col in df.columns if col.startswith('LFQ intensity') and col.endswith('L')]
        else:
            lysis_columns = [col for col in df.columns if col.startswith(norm_type) and col.endswith('L')]

        patient_measurements = {}
        patient_cvs = {}

        for col in lysis_columns:
            measurement = row[col]

            # Extract patient number based on normalization type
            if norm_type == 'LFQ':
                patient = col.split(' ')[-1].split('_')[0]
            else:
                patient = col.split(' ')[1].split('_')[0]

            if patient not in patient_measurements:
                patient_measurements[patient] = []

            if measurement > 0:  # Only include non-zero measurements
                patient_measurements[patient].append(measurement)

        # Calculate CVs for patients with multiple measurements
        for patient, measurements in patient_measurements.items():
            if len(measurements) > 2:  # Require at least 3 measurements
                cv = np.std(measurements) / np.mean(measurements)
                patient_cvs[patient] = cv

        avg_cv = np.mean(list(patient_cvs.values())) if patient_cvs else np.nan

        results.append({
            **dict(zip(df.columns[:4], identifiers)),
            'Patient Measurements': patient_measurements,
            'Patient CVs': patient_cvs,
            'Average CV': avg_cv
        })

    return pd.DataFrame(results)


def select_top_20_proteins(csv_file, min_patients=15):
    # Load the protein analysis CSV
    df = pd.read_csv(csv_file)

    # Convert string representations of dicts to actual dicts
    df['Patient CVs'] = df['Patient CVs'].apply(eval)

    # Add patient count column
    df['Patient Count'] = df['Patient CVs'].apply(len)

    # Filter proteins with minimum number of patients
    filtered_df = df[df['Patient Count'] >= min_patients]

    # Sort by Average CV and select top 20
    top_20_proteins = filtered_df.sort_values('Average CV').head(20)

    return top_20_proteins


def convert_cv_values(input_file, output_file):
    df = pd.read_csv(input_file)
    df['Patient CVs'] = df['Patient CVs'].apply(eval)
    df['Patient CVs'] = df['Patient CVs'].apply(lambda d: {k: v / 100 for k, v in d.items()})
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    base_dir = r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\Thesis"
    csv_base_dir = join(base_dir, "CSVs")
    protein_analysis_csvs_dir = join(csv_base_dir, "protein_analysis_csvs_dir")
    relevant_dataframes_per_norm_type_dir = join(csv_base_dir, "relevant_dataframes_per_norm_type")
    top_20_proteins_per_norm_type_dir = join(csv_base_dir, "top_20_proteins_per_norm_type")

    for norm_type in ['Intensity', 'iBAQ', 'LFQ']:
        relevant_patients_csv_file = f'relevant_patients_proteomics_table_{norm_type}.csv'

        protein_analysis_df = create_protein_analysis_dataframe(
            join(relevant_dataframes_per_norm_type_dir, relevant_patients_csv_file), norm_type)

        protein_analysis_df.to_csv(join(protein_analysis_csvs_dir, f'{norm_type}_protein_analysis.csv'), index=False)

        print(f"{norm_type} analysis complete. Shape: {protein_analysis_df.shape}")

        top_20_df = select_top_20_proteins(join(protein_analysis_csvs_dir, f'{norm_type}_protein_analysis.csv'))
        top_20_df.to_csv(join(top_20_proteins_per_norm_type_dir, f'top_20_proteins_{norm_type}.csv'), index=False)

        print(f"Top 20 proteins for {norm_type}: {top_20_df.shape[0]}")

        convert_cv_values(join(top_20_proteins_per_norm_type_dir, f'top_20_proteins_{norm_type}.csv'),
                          join(top_20_proteins_per_norm_type_dir, f'top_20_proteins_{norm_type}_raw_cv.csv'))
