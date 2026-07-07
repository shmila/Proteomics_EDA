"""
Generate a 3-panel expression distribution plot for a single protein,
showing per-patient measurements under all three normalization types
(Intensity, iBAQ, LFQ).

Set the active protein by (un)commenting one PROTEIN_NAME line in main().
"""

import os
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from config import CSVS_DIR


def load_protein_data(csv_dir):
    """Load the pre-filtered proteomics tables for all three normalization types."""
    data = {}
    for norm_type in ['Intensity', 'iBAQ', 'LFQ']:
        filepath = join(csv_dir, f'relevant_patients_proteomics_table_{norm_type}.csv')
        data[norm_type] = pd.read_csv(filepath)
    return data


def get_protein_row(protein_name, df):
    mask = df['Protein names'] == protein_name
    if not mask.any():
        mask = df['Protein names'].str.contains(protein_name.strip(), na=False, case=False)
    return df[mask].iloc[0] if mask.any() else None


def extract_patient_measurements(protein_row, df, norm_type):
    if norm_type == 'LFQ':
        lysis_cols = [c for c in df.columns if c.startswith('LFQ intensity') and c.endswith('L')]
    else:
        lysis_cols = [c for c in df.columns if c.startswith(norm_type) and c.endswith('L')]

    patient_measurements = {}
    for col in lysis_cols:
        if norm_type == 'LFQ':
            patient = col.split(' ')[-1].split('_')[0]
        else:
            patient = col.split(' ')[1].split('_')[0]
        val = protein_row[col]
        if patient not in patient_measurements:
            patient_measurements[patient] = []
        patient_measurements[patient].append(val)
    return patient_measurements


def plot_protein_expression(protein_name, data_dict, output_dir=None):
    """Create a 3-panel expression distribution plot (one per normalization type)."""
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))
    display_name = protein_name.strip()
    fig.suptitle(f'Expression Levels for {display_name} Across Patients',
                 fontsize=22, fontweight='bold')

    for idx, norm_type in enumerate(['Intensity', 'iBAQ', 'LFQ']):
        df = data_dict[norm_type]
        protein_row = get_protein_row(protein_name, df)

        if protein_row is None:
            print(f"Protein '{protein_name}' not found in {norm_type} data")
            continue

        patient_measurements = extract_patient_measurements(protein_row, df, norm_type)

        all_points = []
        for patient in sorted(patient_measurements.keys(), key=int):
            for i, val in enumerate(patient_measurements[patient]):
                all_points.append((patient, i, val))

        x_coords = list(range(len(all_points)))
        y_values = [p[2] for p in all_points]

        patient_means = []
        patient_ranges = []
        patient_stats = {}
        patient_tick_positions = []
        patient_tick_labels = []
        curr_idx = 0

        for patient in sorted(patient_measurements.keys(), key=int):
            measurements = patient_measurements[patient]
            mean_val = np.mean(measurements)
            std_val = np.std(measurements)
            n = len(measurements)
            cv = (std_val / mean_val) * 100 if mean_val != 0 else 0

            patient_means.append(mean_val)
            start = curr_idx
            end = curr_idx + n - 1
            patient_ranges.append((start, end))
            patient_stats[patient] = {'mean': mean_val, 'std': std_val, 'cv': cv, 'n': n}

            patient_tick_positions.append((start + end) / 2)
            patient_tick_labels.append(patient)

            curr_idx += n

        valid_means = [m for m in patient_means if m > 0]
        cv_of_means = (np.std(valid_means) / np.mean(valid_means)) * 100 if valid_means else 0

        ax = axes[idx]
        ax.scatter(x_coords, y_values, c='blue', s=60, alpha=0.7)

        for (start, end), patient in zip(patient_ranges, sorted(patient_measurements.keys(), key=int)):
            stats = patient_stats[patient]
            ax.hlines(stats['mean'], start, end, colors='red', linestyles='--', alpha=0.7)
            ax.fill_between(range(start, end + 1),
                            [stats['mean'] - stats['std']] * (end - start + 1),
                            [stats['mean'] + stats['std']] * (end - start + 1),
                            color='red', alpha=0.1)

        ax.axhline(y=np.mean(y_values), color='green', linestyle='-', alpha=0.5)

        ax.text(0.02, 0.98, f'CV of patient means: {cv_of_means:.2f}%',
                transform=ax.transAxes, va='top', fontsize=16,
                bbox=dict(facecolor='white', alpha=0.8))

        ax.set_xticks(patient_tick_positions)
        ax.set_xticklabels(patient_tick_labels, fontsize=14)
        ax.set_xlabel('Patient', fontsize=16)
        ax.set_title(f'{norm_type} Normalization', fontsize=20)
        ax.set_ylabel('Expression Level', fontsize=16)
        ax.tick_params(axis='y', labelsize=14)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(14)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_name = display_name.replace(' ', '_').replace(';', '').replace('/', '_')
        filepath = join(output_dir, f'expression_distribution_{safe_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f'Saved to: {filepath}')

    plt.close()
    return fig


def main(protein_name, output_dir=None):
    csv_dir = str(CSVS_DIR / 'relevant_dataframes_per_norm_type')
    data_dict = load_protein_data(csv_dir)
    plot_protein_expression(protein_name, data_dict, output_dir=output_dir)


if __name__ == '__main__':
    # PROTEIN_NAME = " Thioredoxin"
    # PROTEIN_NAME = " Annexin A7"
    # PROTEIN_NAME = " Gelsolin"
    # PROTEIN_NAME = " Actin, cytoplasmic 2"
    # PROTEIN_NAME = " Cornulin"
    # PROTEIN_NAME = " Proliferation marker protein Ki-67"
    # PROTEIN_NAME = " 14-3-3 protein beta/alpha"
    # PROTEIN_NAME = " Ubiquitin-conjugating enzyme E2 L3"
    PROTEIN_NAME = " Elongation factor 1-alpha 1"

    OUTPUT_DIR = None  # set to a path to save the figure

    main(PROTEIN_NAME, output_dir=OUTPUT_DIR)
