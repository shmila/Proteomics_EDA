from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json


def get_latest_multi_run_eval(dataset_dir):
    """Find the latest multi-run evaluation results at dataset level"""
    multi_run_eval_dirs = list(dataset_dir.glob("multi_run_evaluation_*"))
    if not multi_run_eval_dirs:
        print(f"No multi-run evaluation directories found in {dataset_dir}")
        return None

    latest_multi_run = max(multi_run_eval_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Found latest multi-run evaluation: {latest_multi_run}")
    return latest_multi_run


def create_all_runs_plots(slides_df, tiles_df, metrics, output_dir):
    """Create plots aggregating results across all runs from all folds"""

    # 1. All-runs correlation plots (one point per run per slide)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Expression vs Positive Ratio
    for slide_id in slides_df['slide_id'].unique():
        slide_data = slides_df[slides_df['slide_id'] == slide_id]
        ax1.scatter(slide_data['expression_value'], slide_data['positive_ratio'],
                    alpha=0.2, label=f'Slide {slide_id}' if len(ax1.get_legend_handles_labels()[1]) == 0 else "")

    corr = metrics['positive_ratio_correlations']
    ax1.set_title(f'Expression vs Positive Tiles Ratio (All Runs)\n' +
                  f'r={corr["pearson_r"]:.3f} (right-tailed p={corr["pearson_p"]:.1e})\n' +
                  f'ρ={corr["spearman_r"]:.3f} (right-tailed p={corr["spearman_p"]:.1e})')
    ax1.set_xlabel('Expression Value')
    ax1.set_ylabel('Ratio of Positive Tiles')

    # Expression vs Mean Prediction
    for slide_id in slides_df['slide_id'].unique():
        slide_data = slides_df[slides_df['slide_id'] == slide_id]
        ax2.scatter(slide_data['expression_value'], slide_data['mean_prediction'],
                    alpha=0.2, label=f'Slide {slide_id}' if len(ax2.get_legend_handles_labels()[1]) == 0 else "")

    corr = metrics['mean_prediction_correlations']
    ax2.set_title(f'Expression vs Mean Prediction (All Runs)\n' +
                  f'r={corr["pearson_r"]:.3f} (right-tailed p={corr["pearson_p"]:.1e})\n' +
                  f'ρ={corr["spearman_r"]:.3f} (right-tailed p={corr["spearman_p"]:.1e})')
    ax2.set_xlabel('Expression Value')
    ax2.set_ylabel('Mean Prediction')

    plt.tight_layout()
    plt.savefig(output_dir / "all_runs_correlation_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Distribution of correlations across all runs
    run_correlations = []
    for fold in slides_df['fold'].unique():
        for run in slides_df[slides_df['fold'] == fold]['run'].unique():
            run_data = slides_df[(slides_df['fold'] == fold) & (slides_df['run'] == run)]

            # Calculate correlations and convert p-values to right-tailed
            pearson_r, pearson_p_two = stats.pearsonr(run_data['expression_value'], run_data['positive_ratio'])
            pearson_mean_r, pearson_mean_p_two = stats.pearsonr(run_data['expression_value'],
                                                                run_data['mean_prediction'])
            spearman_r, spearman_p_two = stats.spearmanr(run_data['expression_value'], run_data['positive_ratio'])
            spearman_mean_r, spearman_mean_p_two = stats.spearmanr(run_data['expression_value'],
                                                                   run_data['mean_prediction'])

            # Convert to right-tailed p-values
            pearson_ratio_p = pearson_p_two / 2 if pearson_r > 0 else 1 - (pearson_p_two / 2)
            pearson_mean_p = pearson_mean_p_two / 2 if pearson_mean_r > 0 else 1 - (pearson_mean_p_two / 2)
            spearman_ratio_p = spearman_p_two / 2 if spearman_r > 0 else 1 - (spearman_p_two / 2)
            spearman_mean_p = spearman_mean_p_two / 2 if spearman_mean_r > 0 else 1 - (spearman_mean_p_two / 2)

            run_correlations.append({
                'fold': fold,
                'run': run,
                'pearson_ratio': pearson_r,
                'pearson_ratio_p': pearson_ratio_p,
                'pearson_mean': pearson_mean_r,
                'pearson_mean_p': pearson_mean_p,
                'spearman_ratio': spearman_r,
                'spearman_ratio_p': spearman_ratio_p,
                'spearman_mean': spearman_mean_r,
                'spearman_mean_p': spearman_mean_p
            })

    run_corr_df = pd.DataFrame(run_correlations)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    sns.histplot(data=run_corr_df, x='pearson_ratio', bins=20, ax=ax1)
    ax1.set_title('Distribution of Pearson r (Positive Ratio)')
    ax1.set_xlabel('Correlation Coefficient')

    sns.histplot(data=run_corr_df, x='pearson_mean', bins=20, ax=ax2)
    ax2.set_title('Distribution of Pearson r (Mean Prediction)')
    ax2.set_xlabel('Correlation Coefficient')

    sns.histplot(data=run_corr_df, x='spearman_ratio', bins=20, ax=ax3)
    ax3.set_title('Distribution of Spearman ρ (Positive Ratio)')
    ax3.set_xlabel('Correlation Coefficient')

    sns.histplot(data=run_corr_df, x='spearman_mean', bins=20, ax=ax4)
    ax4.set_title('Distribution of Spearman ρ (Mean Prediction)')
    ax4.set_xlabel('Correlation Coefficient')

    plt.tight_layout()
    plt.savefig(output_dir / "all_runs_correlation_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot p-value distributions
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    sns.histplot(data=run_corr_df, x='pearson_ratio_p', bins=20, ax=ax1)
    ax1.set_title('Distribution of Right-Tailed P-Values\nPearson (Positive Ratio)')
    ax1.set_xlabel('p-value')

    sns.histplot(data=run_corr_df, x='pearson_mean_p', bins=20, ax=ax2)
    ax2.set_title('Distribution of Right-Tailed P-Values\nPearson (Mean Prediction)')
    ax2.set_xlabel('p-value')

    sns.histplot(data=run_corr_df, x='spearman_ratio_p', bins=20, ax=ax3)
    ax3.set_title('Distribution of Right-Tailed P-Values\nSpearman (Positive Ratio)')
    ax3.set_xlabel('p-value')

    sns.histplot(data=run_corr_df, x='spearman_mean_p', bins=20, ax=ax4)
    ax4.set_title('Distribution of Right-Tailed P-Values\nSpearman (Mean Prediction)')
    ax4.set_xlabel('p-value')

    plt.tight_layout()
    plt.savefig(output_dir / "all_runs_pvalue_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save correlation statistics
    run_stats = {
        'pearson_ratio': {
            'r': {
                'mean': float(run_corr_df['pearson_ratio'].mean()),
                'std': float(run_corr_df['pearson_ratio'].std()),
                'min': float(run_corr_df['pearson_ratio'].min()),
                'max': float(run_corr_df['pearson_ratio'].max())
            },
            'p_value': {
                'mean': float(run_corr_df['pearson_ratio_p'].mean()),
                'std': float(run_corr_df['pearson_ratio_p'].std()),
                'min': float(run_corr_df['pearson_ratio_p'].min()),
                'max': float(run_corr_df['pearson_ratio_p'].max())
            }
        },
        'pearson_mean': {
            'r': {
                'mean': float(run_corr_df['pearson_mean'].mean()),
                'std': float(run_corr_df['pearson_mean'].std()),
                'min': float(run_corr_df['pearson_mean'].min()),
                'max': float(run_corr_df['pearson_mean'].max())
            },
            'p_value': {
                'mean': float(run_corr_df['pearson_mean_p'].mean()),
                'std': float(run_corr_df['pearson_mean_p'].std()),
                'min': float(run_corr_df['pearson_mean_p'].min()),
                'max': float(run_corr_df['pearson_mean_p'].max())
            }
        },
        'spearman_ratio': {
            'r': {
                'mean': float(run_corr_df['spearman_ratio'].mean()),
                'std': float(run_corr_df['spearman_ratio'].std()),
                'min': float(run_corr_df['spearman_ratio'].min()),
                'max': float(run_corr_df['spearman_ratio'].max())
            },
            'p_value': {
                'mean': float(run_corr_df['spearman_ratio_p'].mean()),
                'std': float(run_corr_df['spearman_ratio_p'].std()),
                'min': float(run_corr_df['spearman_ratio_p'].min()),
                'max': float(run_corr_df['spearman_ratio_p'].max())
            }
        },
        'spearman_mean': {
            'r': {
                'mean': float(run_corr_df['spearman_mean'].mean()),
                'std': float(run_corr_df['spearman_mean'].std()),
                'min': float(run_corr_df['spearman_mean'].min()),
                'max': float(run_corr_df['spearman_mean'].max())
            },
            'p_value': {
                'mean': float(run_corr_df['spearman_mean_p'].mean()),
                'std': float(run_corr_df['spearman_mean_p'].std()),
                'min': float(run_corr_df['spearman_mean_p'].min()),
                'max': float(run_corr_df['spearman_mean_p'].max())
            }
        }
    }

    with open(output_dir / "all_runs_correlation_stats.json", 'w') as f:
        json.dump(run_stats, f, indent=4)

    # 3. Box plots of predictions by true label across all runs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.boxplot(data=slides_df, x='true_label', y='positive_ratio', ax=ax1)
    ax1.set_title('Positive Ratio Distribution by True Label (All Runs)')
    ax1.set_xlabel('True Label')
    ax1.set_ylabel('Ratio of Positive Tiles')

    sns.boxplot(data=slides_df, x='true_label', y='mean_prediction', ax=ax2)
    ax2.set_title('Mean Prediction Distribution by True Label (All Runs)')
    ax2.set_xlabel('True Label')
    ax2.set_ylabel('Mean Prediction Score')

    plt.tight_layout()
    plt.savefig(output_dir / "all_runs_prediction_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()


def aggregate_cv_evaluation_results(dataset_dir):
    """
    Aggregate evaluation results from all CV folds.

    Args:
        dataset_dir: Base directory containing CV fold results
    """
    dataset_dir = Path(dataset_dir)
    print(f"Aggregating results from: {dataset_dir}")

    # Find latest multi-run evaluation directory
    latest_eval_dir = get_latest_multi_run_eval(dataset_dir)
    if latest_eval_dir is None:
        raise ValueError("No multi-run evaluation directory found")

    print(f"\nUsing evaluation results from: {latest_eval_dir}")

    # Collect all slide metrics from all folds and runs
    all_slide_metrics = []
    all_tile_predictions = []

    for fold_dir in latest_eval_dir.glob("fold_*"):
        fold_idx = int(fold_dir.name.split("_")[1])
        print(f"\nProcessing {fold_dir.name}")

        for run_dir in fold_dir.glob("run_*"):
            run_idx = int(run_dir.name.split("_")[1])
            print(f"Processing run {run_idx}")

            try:
                # Load slide metrics
                slide_metrics = pd.read_csv(run_dir / "slide_metrics.csv")
                slide_metrics['fold'] = fold_idx
                slide_metrics['run'] = run_idx
                all_slide_metrics.append(slide_metrics)

                # Load tile predictions
                tile_preds = pd.read_csv(run_dir / "tile_predictions.csv")
                tile_preds['fold'] = fold_idx
                tile_preds['run'] = run_idx
                all_tile_predictions.append(tile_preds)

            except Exception as e:
                print(f"Error loading results from fold {fold_idx}, run {run_idx}: {str(e)}")
                continue

    if not all_slide_metrics:
        raise ValueError("No valid results found in any fold")

    # Combine all metrics
    combined_slides = pd.concat(all_slide_metrics, ignore_index=True)
    combined_tiles = pd.concat(all_tile_predictions, ignore_index=True)

    # Create output directory for aggregated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = dataset_dir / f"consolidated_evaluation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate and save aggregated metrics
    metrics = calculate_aggregated_metrics(combined_slides, combined_tiles)

    # Create visualizations
    create_aggregated_plots(combined_slides, combined_tiles, metrics, output_dir)

    # Create all-runs plots
    create_all_runs_plots(combined_slides, combined_tiles, metrics, output_dir)

    # Save results
    combined_slides.to_csv(output_dir / "all_slide_metrics.csv", index=False)
    combined_tiles.to_csv(output_dir / "all_tile_predictions.csv", index=False)

    with open(output_dir / "aggregated_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\nResults saved to: {output_dir}")
    return metrics, combined_slides, combined_tiles


def calculate_aggregated_metrics(slides_df, tiles_df):
    """Calculate comprehensive metrics from all data"""
    metrics = {}

    # Calculate correlations
    for pred_type in ['positive_ratio', 'mean_prediction']:
        # Calculate two-tailed p-value first
        pearson_r, pearson_p_two = stats.pearsonr(
            slides_df['expression_value'],
            slides_df[pred_type]
        )
        spearman_r, spearman_p_two = stats.spearmanr(
            slides_df['expression_value'],
            slides_df[pred_type]
        )

        # Convert to right-tailed p-values
        pearson_p = pearson_p_two / 2 if pearson_r > 0 else 1 - (pearson_p_two / 2)
        spearman_p = spearman_p_two / 2 if spearman_r > 0 else 1 - (spearman_p_two / 2)

        metrics[f'{pred_type}_correlations'] = {
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p)
        }

    # Calculate per-fold metrics
    fold_metrics = []
    for fold in slides_df['fold'].unique():
        fold_slides = slides_df[slides_df['fold'] == fold]
        fold_tiles = tiles_df[tiles_df['fold'] == fold]

        # Calculate fold-specific correlations
        pearson_ratio = stats.pearsonr(fold_slides['expression_value'],
                                       fold_slides['positive_ratio'])
        pearson_mean = stats.pearsonr(fold_slides['expression_value'],
                                      fold_slides['mean_prediction'])

        fold_metrics.append({
            'fold': int(fold),
            'n_slides': len(fold_slides),
            'n_tiles': len(fold_tiles),
            'positive_ratio_pearson_r': float(pearson_ratio[0]),
            'positive_ratio_pearson_p': float(pearson_ratio[1]),
            'mean_prediction_pearson_r': float(pearson_mean[0]),
            'mean_prediction_pearson_p': float(pearson_mean[1])
        })

    metrics['per_fold'] = fold_metrics

    # Calculate patient-level accuracy
    patient_preds = slides_df.groupby('patient_id').agg({
        'predicted_label': 'mean',
        'true_label': 'first'
    })
    patient_preds['patient_prediction'] = patient_preds['predicted_label'] > 0.5

    # Calculate accuracies
    tile_predictions = (tiles_df['tile_prediction'] > 0.5).astype(int)
    tile_accuracy = np.mean((tile_predictions == tiles_df['true_label']).astype(int))

    slide_predictions = (slides_df['mean_prediction'] > 0.5).astype(int)
    slide_accuracy = np.mean((slide_predictions == slides_df['true_label']).astype(int))

    patient_predictions = (patient_preds['patient_prediction']).astype(int)
    patient_accuracy = np.mean((patient_predictions == patient_preds['true_label']).astype(int))

    # Overall statistics
    metrics['overall'] = {
        'n_folds': len(fold_metrics),
        'total_slides': len(slides_df),
        'total_tiles': len(tiles_df),
        'unique_patients': len(slides_df['patient_id'].unique()),
        'tile_accuracy': float(tile_accuracy),
        'slide_accuracy': float(slide_accuracy),
        'patient_accuracy': float(patient_accuracy)
    }

    # Add support metrics
    metrics['support'] = {
        'tiles': {
            'total': len(tiles_df),
            'positive': int(tiles_df['true_label'].sum()),
            'negative': int(len(tiles_df) - tiles_df['true_label'].sum())
        },
        'slides': {
            'total': len(slides_df),
            'positive': int(slides_df['true_label'].sum()),
            'negative': int(len(slides_df) - slides_df['true_label'].sum())
        },
        'patients': {
            'total': len(patient_preds),
            'positive': int(patient_preds['true_label'].sum()),
            'negative': int(len(patient_preds) - patient_preds['true_label'].sum())
        }
    }

    return metrics


def create_aggregated_plots(slides_df, tiles_df, metrics, output_dir):
    """Create comprehensive visualization of aggregated results"""
    # Set style
    plt.style.use('default')

    # 1. Correlation plots with patient coloring
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Expression vs Positive Ratio
    corr = metrics['positive_ratio_correlations']
    for patient_id in slides_df['patient_id'].unique():
        patient_data = slides_df[slides_df['patient_id'] == patient_id]
        ax1.scatter(patient_data['expression_value'], patient_data['positive_ratio'],
                    alpha=0.5, label=f'Patient {patient_id}')

    ax1.set_title(f'Expression vs Positive Tiles Ratio\n' +
                  f'r={corr["pearson_r"]:.3f} (p={corr["pearson_p"]:.1e})\n' +
                  f'ρ={corr["spearman_r"]:.3f} (p={corr["spearman_p"]:.1e})')
    ax1.set_xlabel('Expression Value')
    ax1.set_ylabel('Ratio of Positive Tiles')

    # Expression vs Mean Prediction
    corr = metrics['mean_prediction_correlations']
    for patient_id in slides_df['patient_id'].unique():
        patient_data = slides_df[slides_df['patient_id'] == patient_id]
        ax2.scatter(patient_data['expression_value'], patient_data['mean_prediction'],
                    alpha=0.5, label=f'Patient {patient_id}')

    ax2.set_title(f'Expression vs Mean Prediction\n' +
                  f'r={corr["pearson_r"]:.3f} (p={corr["pearson_p"]:.1e})\n' +
                  f'ρ={corr["spearman_r"]:.3f} (p={corr["spearman_p"]:.1e})')
    ax2.set_xlabel('Expression Value')
    ax2.set_ylabel('Mean Prediction')

    plt.tight_layout()
    plt.savefig(output_dir / "aggregated_correlation_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Per-fold correlation distributions
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    fold_data = pd.DataFrame(metrics['per_fold'])

    sns.boxplot(data=fold_data, y='positive_ratio_pearson_r', ax=ax1)
    ax1.set_title('Positive Ratio Correlation by Fold')
    ax1.set_ylabel('Pearson r')

    sns.boxplot(data=fold_data, y='positive_ratio_pearson_p', ax=ax2)
    ax2.set_title('Positive Ratio P-value by Fold')
    ax2.set_ylabel('P-value')

    sns.boxplot(data=fold_data, y='mean_prediction_pearson_r', ax=ax3)
    ax3.set_title('Mean Prediction Correlation by Fold')
    ax3.set_ylabel('Pearson r')

    sns.boxplot(data=fold_data, y='mean_prediction_pearson_p', ax=ax4)
    ax4.set_title('Mean Prediction P-value by Fold')
    ax4.set_ylabel('P-value')

    plt.tight_layout()
    plt.savefig(output_dir / "fold_correlation_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Prediction distributions by level
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Tile predictions
    sns.histplot(
        data=tiles_df,
        x='tile_prediction',
        hue='true_label',
        bins=50,
        ax=ax1,
        alpha=0.6,
        legend=True
    )
    ax1.set_title('Distribution of Tile Predictions')
    ax1.set_xlabel('Prediction Score')

    # Slide predictions
    sns.histplot(
        data=slides_df,
        x='mean_prediction',
        hue='true_label',
        bins=50,
        ax=ax2,
        alpha=0.6,
        legend=True
    )
    ax2.set_title('Distribution of Slide Mean Predictions')
    ax2.set_xlabel('Mean Prediction Score')

    plt.tight_layout()
    plt.savefig(output_dir / "prediction_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Expression value distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=slides_df, x='expression_value', bins=30)
    plt.title('Distribution of Expression Values')
    plt.xlabel('Expression Value')
    plt.ylabel('Count')
    plt.savefig(output_dir / "expression_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Box plots of predictions by true label
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.boxplot(data=slides_df, x='true_label', y='positive_ratio', ax=ax1)
    ax1.set_title('Positive Ratio Distribution by True Label')
    ax1.set_xlabel('True Label')
    ax1.set_ylabel('Ratio of Positive Tiles')

    sns.boxplot(data=slides_df, x='true_label', y='mean_prediction', ax=ax2)
    ax2.set_title('Mean Prediction Distribution by True Label')
    ax2.set_xlabel('True Label')
    ax2.set_ylabel('Mean Prediction Score')

    plt.tight_layout()
    plt.savefig(output_dir / "prediction_boxplots.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Configuration
    THESIS_DIR = Path(r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\Thesis")
    DATASET_DIR = THESIS_DIR / "dataset"

    # PROTEIN_NAME = " Thioredoxin"
    # PROTEIN_NAME = " Elongation factor 1-alpha 1"
    # PROTEIN_NAME = " Ubiquitin-conjugating enzyme E2 L3"
    # NORM_TYPE = "Intensity"
    # PROTEIN_NAME = " Actin, cytoplasmic 2"
    # PROTEIN_NAME = " Annexin A7"
    # PROTEIN_NAME = " Gelsolin"
    # PROTEIN_NAME = " 14-3-3 protein beta_alpha"
    # PROTEIN_NAME = " Cornulin"
    PROTEIN_NAME = " Proliferation marker protein Ki-67"
    NORM_TYPE = "LFQ"

    dataset_path = DATASET_DIR / f"tiles_dataset_100x100_{PROTEIN_NAME}_{NORM_TYPE}_cv"

    try:
        metrics, slides_df, tiles_df = aggregate_cv_evaluation_results(dataset_path)

        print("\nAggregated Metrics Summary:")
        print(f"Total Slides: {metrics['overall']['total_slides']}")
        print(f"Total Tiles: {metrics['overall']['total_tiles']}")
        print(f"Unique Patients: {metrics['overall']['unique_patients']}")

        print("\nSupport:")
        for level in ['tiles', 'slides', 'patients']:
            support = metrics['support'][level]
            print(f"\n{level.capitalize()}:")
            print(f"Total: {support['total']}")
            print(f"Positive: {support['positive']}")
            print(f"Negative: {support['negative']}")

        print("\nAccuracies:")
        print(f"Tile-level: {metrics['overall']['tile_accuracy']:.3f}")
        print(f"Slide-level: {metrics['overall']['slide_accuracy']:.3f}")
        print(f"Patient-level: {metrics['overall']['patient_accuracy']:.3f}")

        print("\nCorrelations:")
        for pred_type in ['positive_ratio', 'mean_prediction']:
            corr = metrics[f'{pred_type}_correlations']
            print(f"\n{pred_type}:")
            print(f"Pearson r: {corr['pearson_r']:.3f} (p={corr['pearson_p']:.1e})")
            print(f"Spearman ρ: {corr['spearman_r']:.3f} (p={corr['spearman_p']:.1e})")

        print("\nPer-fold correlations:")
        for fold_metric in metrics['per_fold']:
            fold = fold_metric['fold']
            print(f"\nFold {fold}:")
            print(
                f"Positive ratio - r: {fold_metric['positive_ratio_pearson_r']:.3f} (p={fold_metric['positive_ratio_pearson_p']:.1e})")
            print(
                f"Mean prediction - r: {fold_metric['mean_prediction_pearson_r']:.3f} (p={fold_metric['mean_prediction_pearson_p']:.1e})")

    except Exception as e:
        print(f"Error during aggregation: {str(e)}")
        raise
