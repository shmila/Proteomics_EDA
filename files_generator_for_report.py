import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def generate_report_materials(norm_types=['Intensity', 'iBAQ', 'LFQ']):
    # Create directory if it doesn't exist
    output_dir = 'files_for_claude_to_make_report'
    os.makedirs(output_dir, exist_ok=True)

    # Rest of the code stays the same, just modify save paths
    for norm_type in norm_types:
        df = pd.read_csv(f'tumor_protein_analysis_{norm_type}.csv')
        top_20_df = pd.read_csv(f'tumor_top_20_proteins_{norm_type}.csv')

        df['Patient CVs'] = df['Patient CVs'].apply(eval)
        df = df[df['Patient CVs'].apply(len) >= 2]

        # Distribution Plot
        plt.figure(figsize=(12, 6))
        plt.hist(df[~df['Protein IDs'].isin(top_20_df['Protein IDs'])]['Average CV'],
                 bins=5800, color='blue', alpha=0.7,
                 label=f'Other Proteins (n={len(df) - 20})')
        plt.hist(top_20_df['Average CV'], bins=20, color='red', alpha=0.7,
                 label='Top 20 Most Stable Proteins')
        plt.xlabel('Average Coefficient of Variation (CV)')
        plt.ylabel('Number of Proteins')
        plt.title(f'Distribution of Protein Expression Variability\n{norm_type} Normalization')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'report_distribution_{norm_type}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Top 20 Table
        report_table = top_20_df[['Protein names', 'Gene names', 'Average CV']].copy()
        report_table['Patient Count'] = top_20_df['Patient CVs'].apply(lambda x: len(eval(x)))
        report_table.to_csv(os.path.join(output_dir, f'report_top20_table_{norm_type}.csv'),
                            index=False)

        # CV Boxplot
        cv_data = []
        for protein_data in top_20_df['Patient CVs']:
            cv_values = list(eval(protein_data).values())
            cv_data.extend(cv_values)

        plt.figure(figsize=(8, 6))
        plt.boxplot(cv_data)
        plt.title(f'CV Distribution Across Patients\n{norm_type} Top 20 Proteins')
        plt.ylabel('Coefficient of Variation')
        plt.savefig(os.path.join(output_dir, f'report_cv_boxplot_{norm_type}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Method Comparison
    median_cvs = {}
    for norm_type in norm_types:
        df = pd.read_csv(f'tumor_protein_analysis_{norm_type}.csv')
        df['Patient CVs'] = df['Patient CVs'].apply(eval)
        df = df[df['Patient CVs'].apply(len) >= 2]

        median_cvs[norm_type] = {
            'Top 20': df.nsmallest(20, 'Average CV')['Average CV'].median(),
            'Others': df.nlargest(len(df) - 20, 'Average CV')['Average CV'].median()
        }

    comparison_data = pd.DataFrame(median_cvs).T
    comparison_data.to_csv(os.path.join(output_dir, 'report_method_comparison.csv'))

    plt.figure(figsize=(10, 6))
    comparison_data.plot(kind='bar')
    plt.title('Median CV Comparison Across Methods')
    plt.xlabel('Normalization Method')
    plt.ylabel('Median CV')
    plt.legend(title='Protein Group')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'report_method_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    generate_report_materials()
