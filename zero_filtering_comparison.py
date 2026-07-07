"""
Compare protein stability rankings with different relaxation strategies.

Current: filter zeros, require all 3 measurements non-zero.
Relaxed min=1: no zero filtering, require >=1 non-zero measurement (exclude all-zeros patients).
Relaxed min=0: no zero filtering, include everyone (all-zeros patients get CV=0).
"""

import pandas as pd
import numpy as np
from os.path import join

BASE_DIR = r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\Thesis"
CSV_DIR = join(BASE_DIR, "CSVs")
INPUT_DIR = join(CSV_DIR, "relevant_dataframes_per_norm_type")


def get_lysis_columns(df, norm_type):
    if norm_type == 'LFQ':
        return [col for col in df.columns if col.startswith('LFQ intensity') and col.endswith('L')]
    else:
        return [col for col in df.columns if col.startswith(norm_type) and col.endswith('L')]


def extract_patient(col, norm_type):
    if norm_type == 'LFQ':
        return col.split(' ')[-1].split('_')[0]
    else:
        return col.split(' ')[1].split('_')[0]


def analyze_current(df, norm_type):
    """Current approach: filter zeros, require all 3 non-zero."""
    lysis_columns = get_lysis_columns(df, norm_type)
    results = []

    for _, row in df.iterrows():
        patient_measurements = {}
        for col in lysis_columns:
            patient = extract_patient(col, norm_type)
            if patient not in patient_measurements:
                patient_measurements[patient] = []
            if row[col] > 0:
                patient_measurements[patient].append(row[col])

        patient_cvs = {}
        for patient, measurements in patient_measurements.items():
            if len(measurements) >= 3:
                cv = np.std(measurements) / np.mean(measurements)
                patient_cvs[patient] = cv

        avg_cv = np.mean(list(patient_cvs.values())) if patient_cvs else np.nan

        results.append({
            'Protein names': row['Protein names'],
            'Gene names': row['Gene names'],
            'Average CV': avg_cv,
            'Patient Count': len(patient_cvs),
        })

    return pd.DataFrame(results)


def analyze_no_filter(df, norm_type, min_nonzero):
    """No zero filtering. Use all 3 raw measurements.
    min_nonzero=1: require at least 1 non-zero (exclude all-zeros patients).
    min_nonzero=0: include everyone (all-zeros patients get CV=0).
    """
    lysis_columns = get_lysis_columns(df, norm_type)
    results = []

    for _, row in df.iterrows():
        patient_measurements = {}
        for col in lysis_columns:
            patient = extract_patient(col, norm_type)
            if patient not in patient_measurements:
                patient_measurements[patient] = []
            patient_measurements[patient].append(row[col])

        patient_cvs = {}
        for patient, measurements in patient_measurements.items():
            if len(measurements) >= 3:
                nonzero_count = sum(1 for m in measurements if m > 0)

                if min_nonzero == 0:
                    # Include everyone
                    mean_val = np.mean(measurements)
                    if mean_val == 0:
                        cv = 0.0  # [0,0,0] -> CV = 0
                    else:
                        cv = np.std(measurements) / mean_val
                    patient_cvs[patient] = cv

                elif min_nonzero == 1:
                    # Require at least 1 non-zero
                    if nonzero_count >= 1:
                        cv = np.std(measurements) / np.mean(measurements)
                        patient_cvs[patient] = cv

        avg_cv = np.mean(list(patient_cvs.values())) if patient_cvs else np.nan

        results.append({
            'Protein names': row['Protein names'],
            'Gene names': row['Gene names'],
            'Average CV': avg_cv,
            'Patient Count': len(patient_cvs),
        })

    return pd.DataFrame(results)


def get_top_20(analysis_df, min_patients=15):
    filtered = analysis_df[analysis_df['Patient Count'] >= min_patients].copy()
    return filtered.sort_values('Average CV').head(20).reset_index(drop=True)


def print_comparison(norm_type, current_top20, min1_top20, min0_top20):
    print(f"\n{'='*120}")
    print(f" {norm_type} NORMALIZATION - TOP 20 COMPARISON")
    print(f"{'='*120}")

    print(f"\n{'Rank':<5} | {'CURRENT (3 non-zero, filter zeros)':<40} | {'min=1 (no filter, excl all-zeros)':<40} | {'min=0 (no filter, incl all-zeros)':<40}")
    print(f"{'':5} | {'Protein':<25} {'CV':>8} {'#P':>4} | {'Protein':<25} {'CV':>8} {'#P':>4} | {'Protein':<25} {'CV':>8} {'#P':>4}")
    print(f"{'-'*5}-+-{'-'*40}-+-{'-'*40}-+-{'-'*40}")

    for i in range(20):
        line = f"{i+1:<5}"
        for top20 in [current_top20, min1_top20, min0_top20]:
            if i < len(top20):
                name = top20.iloc[i]['Protein names'].strip()
                name = name[:23] + '..' if len(name) > 25 else name
                cv = f"{top20.iloc[i]['Average CV']:.4f}"
                pat = f"{top20.iloc[i]['Patient Count']}"
            else:
                name, cv, pat = '---', '---', '---'
            line += f" | {name:<25} {cv:>8} {pat:>4}"
        print(line)

    # Overlap
    sets = {
        'current': set(current_top20['Protein names'].str.strip()),
        'min=1': set(min1_top20['Protein names'].str.strip()),
        'min=0': set(min0_top20['Protein names'].str.strip()),
    }

    print(f"\n--- OVERLAP ---")
    for a, b in [('current', 'min=1'), ('current', 'min=0'), ('min=1', 'min=0')]:
        overlap = sets[a] & sets[b]
        print(f"  {a} vs {b}: {len(overlap)}/20 in common")
        if overlap and len(overlap) < 20:
            print(f"    Shared: {', '.join(sorted(overlap))}")
        if len(overlap) < 20:
            only_a = sets[a] - sets[b]
            only_b = sets[b] - sets[a]
            if only_a:
                print(f"    Only in {a}: {', '.join(sorted(only_a))}")
            if only_b:
                print(f"    Only in {b}: {', '.join(sorted(only_b))}")


if __name__ == '__main__':
    for norm_type in ['Intensity', 'iBAQ', 'LFQ']:
        csv_file = join(INPUT_DIR, f'relevant_patients_proteomics_table_{norm_type}.csv')
        df = pd.read_csv(csv_file)

        print(f"\n{'#'*120}")
        print(f" {norm_type}: {len(df)} proteins")
        print(f"{'#'*120}")

        # Run analyses
        current_df = analyze_current(df, norm_type)
        min1_df = analyze_no_filter(df, norm_type, min_nonzero=1)
        min0_df = analyze_no_filter(df, norm_type, min_nonzero=0)

        # Eligible counts
        for label, adf in [('current (3 non-zero)', current_df), ('min=1 (no filter)', min1_df), ('min=0 (no filter)', min0_df)]:
            eligible = adf[adf['Patient Count'] >= 15]
            print(f"  {label}: {len(eligible)} proteins with >=15 patients")

        # Top 20
        current_top20 = get_top_20(current_df)
        min1_top20 = get_top_20(min1_df)
        min0_top20 = get_top_20(min0_df)

        print_comparison(norm_type, current_top20, min1_top20, min0_top20)
