import os
import shutil
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from tqdm import tqdm
from weak_supervision_label_predictor.model.protein_expression_model import TileDataset, ProteinExpressionModel
from torch.utils.data import DataLoader


class ProteinExpressionMultiRunEvaluator:
    def __init__(self, model_dir, n_runs=20, n_tiles_per_slide=100,
                 fixed_seeds=None):
        """
        Initialize evaluator for multiple test set runs.

        Args:
            model_dir: Directory containing trained model
            n_runs: Number of evaluation runs per fold
            n_tiles_per_slide: Number of tiles to sample per slide for each run
            fixed_seeds: List of seeds for reproducible sampling
        """
        self.model_dir = Path(model_dir)
        self.n_runs = n_runs
        self.n_tiles_per_slide = n_tiles_per_slide

        # Set fixed seeds if not provided
        if fixed_seeds is None:
            fixed_seeds = list(range(42, 42 + n_runs))
        self.fixed_seeds = fixed_seeds

        # Load model and configuration
        self.load_model()
        print(f"Initialized evaluator for {n_runs} runs with {n_tiles_per_slide} tiles per slide")

    def load_model(self):
        """Load trained model and configuration"""
        config_path = self.model_dir / "training_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        model_path = self.model_dir / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path)
        self.model = ProteinExpressionModel(model_type=checkpoint['model_type'])
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.model.eval()

    def evaluate_single_run(self, test_dataset, run_idx, seed, output_dir):
        """
        Evaluate model on a single random sampling of test set tiles.

        Args:
            test_dataset: Full test dataset
            run_idx: Current run index
            seed: Random seed for sampling
            output_dir: Directory to save results

        Returns:
            Dictionary containing run metrics and correlation results
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create sampled test dataset
        sampled_tiles = self._sample_test_tiles(test_dataset)
        test_loader = DataLoader(sampled_tiles, batch_size=32, shuffle=False)

        # Collect predictions
        predictions = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Run {run_idx} evaluation"):
                images = batch['image'].to(self.model.device)
                outputs = self.model.model(images).squeeze()

                for i in range(len(outputs)):
                    predictions.append({
                        'tile_prediction': outputs[i].item(),
                        'slide_id': batch['slide_id'][i],
                        'patient_id': batch['patient_id'][i],
                        'true_label': batch['label'][i].item(),
                        'expression_value': batch['expression_value'][i].item()
                    })

        # Convert to DataFrame and calculate slide-level metrics
        results_df = pd.DataFrame(predictions)
        slide_metrics = self._calculate_slide_metrics(results_df)

        # Calculate correlations
        corr_metrics = self._calculate_correlations(slide_metrics)

        # Save run results
        run_dir = output_dir / f"run_{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(run_dir / "tile_predictions.csv", index=False)
        slide_metrics.to_csv(run_dir / "slide_metrics.csv", index=False)

        with open(run_dir / "correlation_metrics.json", 'w') as f:
            json.dump(corr_metrics, f, indent=4)

        # Create run visualizations
        print(f"\nCreating visualizations for run {run_idx}")
        print(f"Run directory: {run_dir}")
        print(f"Run directory exists: {run_dir.exists()}")
        self._create_run_plots(results_df, slide_metrics, corr_metrics, run_dir)

        return {
            'run_idx': run_idx,
            'seed': seed,
            'slide_metrics': slide_metrics,
            'correlations': corr_metrics
        }

    def evaluate_fold(self, fold_idx, test_dataset, output_dir):
        """
        Evaluate model multiple times on a fold's test set.

        Args:
            fold_idx: Current fold index
            test_dataset: Test dataset for this fold
            output_dir: Base directory for evaluation results

        Returns:
            Dictionary containing aggregated fold results
        """
        print(f"\nEvaluating fold {fold_idx} with {self.n_runs} runs")
        print(f"Number of fixed seeds: {len(self.fixed_seeds)}")
        print(f"Fixed seeds: {self.fixed_seeds}")

        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_results = []
        for run_idx, seed in enumerate(self.fixed_seeds):
            run_results = self.evaluate_single_run(
                test_dataset, run_idx, seed, fold_dir
            )
            fold_results.append(run_results)

        # Aggregate results across runs
        aggregated_results = self._aggregate_fold_results(fold_results)

        # Save aggregated results
        with open(fold_dir / "aggregated_fold_results.json", 'w') as f:
            json.dump(aggregated_results, f, indent=4)

        # Create fold-level visualizations
        self._create_fold_plots(fold_results, fold_dir)

        return aggregated_results

    def _sample_test_tiles(self, test_dataset):
        """Sample fixed number of tiles per slide from test dataset"""
        slide_groups = {}
        for idx in range(len(test_dataset)):
            tile_info = test_dataset.tiles[idx]
            slide_id = tile_info['slide_id']
            if slide_id not in slide_groups:
                slide_groups[slide_id] = []
            slide_groups[slide_id].append(idx)

        sampled_indices = []
        for slide_indices in slide_groups.values():
            if len(slide_indices) <= self.n_tiles_per_slide:
                sampled_indices.extend(slide_indices)
            else:
                sampled = np.random.choice(
                    slide_indices,
                    size=self.n_tiles_per_slide,
                    replace=False
                )
                sampled_indices.extend(sampled)

        return torch.utils.data.Subset(test_dataset, sampled_indices)

    def _calculate_slide_metrics(self, results_df):
        """Calculate metrics at slide level"""
        slide_metrics = []
        for slide_id in results_df['slide_id'].unique():
            slide_data = results_df[results_df['slide_id'] == slide_id]

            # Get ground truth info
            true_label = slide_data['true_label'].iloc[0]
            expression_value = slide_data['expression_value'].iloc[0]
            patient_id = slide_data['patient_id'].iloc[0]

            # Calculate metrics
            n_tiles = len(slide_data)
            n_positive = sum(slide_data['tile_prediction'] > 0.5)
            positive_ratio = n_positive / n_tiles
            mean_prediction = slide_data['tile_prediction'].mean()
            std_prediction = slide_data['tile_prediction'].std()

            slide_metrics.append({
                'slide_id': slide_id,
                'patient_id': patient_id,
                'true_label': true_label,
                'expression_value': expression_value,
                'n_tiles': n_tiles,
                'n_positive_tiles': n_positive,
                'positive_ratio': positive_ratio,
                'mean_prediction': mean_prediction,
                'std_prediction': std_prediction,
                'predicted_label': positive_ratio > 0.5
            })

        return pd.DataFrame(slide_metrics)

    def _calculate_correlations(self, slide_metrics):
        """Calculate correlation metrics"""
        correlations = {}

        for pred_col in ['positive_ratio', 'mean_prediction']:
            pearson_r, pearson_p = stats.pearsonr(
                slide_metrics[pred_col],
                slide_metrics['expression_value']
            )

            spearman_r, spearman_p = stats.spearmanr(
                slide_metrics[pred_col],
                slide_metrics['expression_value']
            )

            correlations[pred_col] = {
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p)
            }

        return correlations

    def _create_run_plots(self, results_df, slide_metrics, corr_metrics, output_dir):
        """Create visualizations for a single evaluation run"""
        print(f"\nIn _create_run_plots")
        print(f"Output directory: {output_dir}")

        # Create temp directory for matplotlib
        temp_dir = Path(THESIS_DIR) / "tmp_plots"
        temp_dir.mkdir(exist_ok=True)

        try:
            # Tile prediction distribution
            print("Creating tile prediction plot...")
            plt.figure(figsize=(10, 6))
            plt.hist([
                results_df[results_df['true_label'] == 0]['tile_prediction'],
                results_df[results_df['true_label'] == 1]['tile_prediction']
            ], label=['Negative', 'Positive'], bins=50, alpha=0.7)
            plt.title('Distribution of Tile Predictions by True Label')
            plt.xlabel('Prediction Score')
            plt.ylabel('Count')
            plt.legend()

            # Save to temp directory first
            temp_path = temp_dir / "temp_dist.png"
            plt.savefig(temp_path)
            plt.close()

            # Move to final location
            final_path = output_dir / "tile_predictions_dist.png"
            shutil.copy2(temp_path, final_path)
            temp_path.unlink()
            print(f"Saved tile distribution plot to: {final_path}")

            # Correlation plots
            print("Creating correlation plots...")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Expression vs Positive Ratio
            ratio_corr = corr_metrics['positive_ratio']
            ax1.scatter(slide_metrics['expression_value'], slide_metrics['positive_ratio'])
            ax1.set_title(f'Expression vs Positive Tiles Ratio\n' +
                          f'r={ratio_corr["pearson_r"]:.3f} (p={ratio_corr["pearson_p"]:.1e})\n' +
                          f'ρ={ratio_corr["spearman_r"]:.3f} (p={ratio_corr["spearman_p"]:.1e})')
            ax1.set_xlabel('Expression Value')
            ax1.set_ylabel('Ratio of Positive Tiles')

            # Expression vs Mean Prediction
            mean_corr = corr_metrics['mean_prediction']
            ax2.scatter(slide_metrics['expression_value'], slide_metrics['mean_prediction'])
            ax2.set_title(f'Expression vs Mean Prediction\n' +
                          f'r={mean_corr["pearson_r"]:.3f} (p={mean_corr["pearson_p"]:.1e})\n' +
                          f'ρ={mean_corr["spearman_r"]:.3f} (p={mean_corr["spearman_p"]:.1e})')
            ax2.set_xlabel('Expression Value')
            ax2.set_ylabel('Mean Prediction')

            plt.tight_layout()

            # Save correlation plots to temp directory
            temp_path = temp_dir / "temp_corr.png"
            plt.savefig(temp_path)
            plt.close()

            # Move to final location
            final_path = output_dir / "correlation_plots.png"
            shutil.copy2(temp_path, final_path)
            temp_path.unlink()
            print(f"Saved correlation plots to: {final_path}")

        except Exception as e:
            print(f"Error creating plots: {str(e)}")
            raise
        finally:
            # Clean up temporary directory
            try:
                if temp_dir.exists():
                    for temp_file in temp_dir.glob("*.png"):
                        temp_file.unlink()
                    temp_dir.rmdir()
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory: {str(e)}")

    def _create_fold_plots(self, fold_results, output_dir):
        """Create visualizations aggregating results across all runs in a fold"""
        # Extract correlation coefficients and p-values
        coeffs = {
            'pearson_ratio': [],
            'pearson_mean': [],
            'spearman_ratio': [],
            'spearman_mean': [],
            'pearson_p_ratio': [],
            'pearson_p_mean': [],
            'spearman_p_ratio': [],
            'spearman_p_mean': []
        }

        for run in fold_results:
            corr = run['correlations']
            coeffs['pearson_ratio'].append(corr['positive_ratio']['pearson_r'])
            coeffs['pearson_mean'].append(corr['mean_prediction']['pearson_r'])
            coeffs['spearman_ratio'].append(corr['positive_ratio']['spearman_r'])
            coeffs['spearman_mean'].append(corr['mean_prediction']['spearman_r'])
            coeffs['pearson_p_ratio'].append(corr['positive_ratio']['pearson_p'])
            coeffs['pearson_p_mean'].append(corr['mean_prediction']['pearson_p'])
            coeffs['spearman_p_ratio'].append(corr['positive_ratio']['spearman_p'])
            coeffs['spearman_p_mean'].append(corr['mean_prediction']['spearman_p'])

        # Plot correlation coefficient distributions
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        sns.histplot(coeffs['pearson_ratio'], ax=ax1)
        ax1.set_title('Pearson r (Positive Ratio)')
        ax1.set_xlabel('Correlation Coefficient')

        sns.histplot(coeffs['pearson_mean'], ax=ax2)
        ax2.set_title('Pearson r (Mean Prediction)')
        ax2.set_xlabel('Correlation Coefficient')

        sns.histplot(coeffs['spearman_ratio'], ax=ax3)
        ax3.set_title('Spearman ρ (Positive Ratio)')
        ax3.set_xlabel('Correlation Coefficient')

        sns.histplot(coeffs['spearman_mean'], ax=ax4)
        ax4.set_title('Spearman ρ (Mean Prediction)')
        ax4.set_xlabel('Correlation Coefficient')

        plt.tight_layout()
        plt.savefig(output_dir / "correlation_distributions.png")
        plt.close()

        # Plot p-value distributions
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        sns.histplot(coeffs['pearson_p_ratio'], ax=ax1)
        ax1.set_title('Pearson p-value (Positive Ratio)')
        ax1.set_xlabel('p-value')

        sns.histplot(coeffs['pearson_p_mean'], ax=ax2)
        ax2.set_title('Pearson p-value (Mean Prediction)')
        ax2.set_xlabel('p-value')

        sns.histplot(coeffs['spearman_p_ratio'], ax=ax3)
        ax3.set_title('Spearman p-value (Positive Ratio)')
        ax3.set_xlabel('p-value')

        sns.histplot(coeffs['spearman_p_mean'], ax=ax4)
        ax4.set_title('Spearman p-value (Mean Prediction)')
        ax4.set_xlabel('p-value')

        plt.tight_layout()
        plt.savefig(output_dir / "pvalue_distributions.png")
        plt.close()

        # Create aggregated correlation plots
        all_slide_data = []
        for run in fold_results:
            slide_data = run['slide_metrics'].copy()
            slide_data['run_idx'] = run['run_idx']
            all_slide_data.append(slide_data)

        all_slide_df = pd.concat(all_slide_data, ignore_index=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Expression vs Positive Ratio (all runs)
        for slide_id in all_slide_df['slide_id'].unique():
            slide_runs = all_slide_df[all_slide_df['slide_id'] == slide_id]
            ax1.scatter(slide_runs['expression_value'], slide_runs['positive_ratio'],
                        alpha=0.3, label=slide_id if len(ax1.get_legend_handles_labels()[1]) == 0 else "")

        pearson_r, pearson_p = stats.pearsonr(all_slide_df['expression_value'],
                                              all_slide_df['positive_ratio'])
        spearman_r, spearman_p = stats.spearmanr(all_slide_df['expression_value'],
                                                 all_slide_df['positive_ratio'])

        ax1.set_title(f'Expression vs Positive Tiles Ratio (All Runs)\n' +
                      f'r={pearson_r:.3f} (p={pearson_p:.1e})\n' +
                      f'ρ={spearman_r:.3f} (p={spearman_p:.1e})')
        ax1.set_xlabel('Expression Value')
        ax1.set_ylabel('Ratio of Positive Tiles')

        # Expression vs Mean Prediction (all runs)
        for slide_id in all_slide_df['slide_id'].unique():
            slide_runs = all_slide_df[all_slide_df['slide_id'] == slide_id]
            ax2.scatter(slide_runs['expression_value'], slide_runs['mean_prediction'],
                        alpha=0.3, label=slide_id if len(ax2.get_legend_handles_labels()[1]) == 0 else "")

        pearson_r, pearson_p = stats.pearsonr(all_slide_df['expression_value'],
                                              all_slide_df['mean_prediction'])
        spearman_r, spearman_p = stats.spearmanr(all_slide_df['expression_value'],
                                                 all_slide_df['mean_prediction'])

        ax2.set_title(f'Expression vs Mean Prediction (All Runs)\n' +
                      f'r={pearson_r:.3f} (p={pearson_p:.1e})\n' +
                      f'ρ={spearman_r:.3f} (p={spearman_p:.1e})')
        ax2.set_xlabel('Expression Value')
        ax2.set_ylabel('Mean Prediction')

        plt.tight_layout()
        plt.savefig(output_dir / "aggregated_correlation_plots.png")
        plt.close()

    def _aggregate_fold_results(self, fold_results):
        """Aggregate statistics and metrics across all runs in a fold"""
        # Extract correlation coefficients and their distributions
        corr_stats = {
            'positive_ratio': {
                'pearson_r': {
                    'mean': np.mean([r['correlations']['positive_ratio']['pearson_r'] for r in fold_results]),
                    'std': np.std([r['correlations']['positive_ratio']['pearson_r'] for r in fold_results]),
                    'distribution': [r['correlations']['positive_ratio']['pearson_r'] for r in fold_results]
                },
                'pearson_p': {
                    'mean': np.mean([r['correlations']['positive_ratio']['pearson_p'] for r in fold_results]),
                    'std': np.std([r['correlations']['positive_ratio']['pearson_p'] for r in fold_results]),
                    'distribution': [r['correlations']['positive_ratio']['pearson_p'] for r in fold_results]
                },
                'spearman_r': {
                    'mean': np.mean([r['correlations']['positive_ratio']['spearman_r'] for r in fold_results]),
                    'std': np.std([r['correlations']['positive_ratio']['spearman_r'] for r in fold_results]),
                    'distribution': [r['correlations']['positive_ratio']['spearman_r'] for r in fold_results]
                },
                'spearman_p': {
                    'mean': np.mean([r['correlations']['positive_ratio']['spearman_p'] for r in fold_results]),
                    'std': np.std([r['correlations']['positive_ratio']['spearman_p'] for r in fold_results]),
                    'distribution': [r['correlations']['positive_ratio']['spearman_p'] for r in fold_results]
                }
            },
            'mean_prediction': {
                'pearson_r': {
                    'mean': np.mean([r['correlations']['mean_prediction']['pearson_r'] for r in fold_results]),
                    'std': np.std([r['correlations']['mean_prediction']['pearson_r'] for r in fold_results]),
                    'distribution': [r['correlations']['mean_prediction']['pearson_r'] for r in fold_results]
                },
                'pearson_p': {
                    'mean': np.mean([r['correlations']['mean_prediction']['pearson_p'] for r in fold_results]),
                    'std': np.std([r['correlations']['mean_prediction']['pearson_p'] for r in fold_results]),
                    'distribution': [r['correlations']['mean_prediction']['pearson_p'] for r in fold_results]
                },
                'spearman_r': {
                    'mean': np.mean([r['correlations']['mean_prediction']['spearman_r'] for r in fold_results]),
                    'std': np.std([r['correlations']['mean_prediction']['spearman_r'] for r in fold_results]),
                    'distribution': [r['correlations']['mean_prediction']['spearman_r'] for r in fold_results]
                },
                'spearman_p': {
                    'mean': np.mean([r['correlations']['mean_prediction']['spearman_p'] for r in fold_results]),
                    'std': np.std([r['correlations']['mean_prediction']['spearman_p'] for r in fold_results]),
                    'distribution': [r['correlations']['mean_prediction']['spearman_p'] for r in fold_results]
                }
            }
        }

        return {
            'n_runs': len(fold_results),
            'correlation_statistics': corr_stats,
            'config': self.config,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


def evaluate_protein_model_cv_multi_run(protein_name, norm_type='Intensity', n_folds=5,
                                        n_runs=20, n_tiles_per_slide=100):
    """Evaluate protein model using cross-validation with multiple test set runs."""
    print(f"\nStarting multi-run evaluation for protein: {protein_name}")
    print(f"Using {n_runs} runs per fold with {n_tiles_per_slide} tiles per slide")

    # Get dataset directory - use original name
    dataset_dir = Path(DATASET_DIR) / f"tiles_dataset_100x100_{protein_name}_{norm_type}_cv"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"CV Dataset directory not found: {dataset_dir}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = dataset_dir / f"multi_run_evaluation_{timestamp}"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluation configuration
    config = {
        'protein_name': protein_name,
        'norm_type': norm_type,
        'n_folds': n_folds,
        'n_runs': n_runs,
        'n_tiles_per_slide': n_tiles_per_slide,
        'timestamp': timestamp
    }

    with open(output_base_dir / "evaluation_config.json", 'w') as f:
        json.dump(config, f, indent=4)

    # Evaluate each fold
    fold_results = []
    for fold in range(n_folds):
        try:
            print(f"\nProcessing fold {fold + 1}/{n_folds}")

            # Get fold directory and latest model
            fold_dir = dataset_dir / f"fold_{fold}"
            model_dir = get_latest_model_dir(fold_dir, protein_name, norm_type)

            # Load test dataset
            labels_df = pd.read_csv(fold_dir / "dataset_info.csv")
            test_df = labels_df[labels_df['split'] == 'test'].copy()
            test_dataset = TileDataset(fold_dir, test_df, split='test')

            # Initialize evaluator and run evaluation
            evaluator = ProteinExpressionMultiRunEvaluator(
                model_dir=model_dir,
                n_runs=n_runs,
                n_tiles_per_slide=n_tiles_per_slide
            )

            fold_result = evaluator.evaluate_fold(
                fold_idx=fold,
                test_dataset=test_dataset,
                output_dir=output_base_dir
            )

            fold_results.append(fold_result)

        except Exception as e:
            print(f"Error processing fold {fold}: {str(e)}")
            continue

    if not fold_results:
        raise ValueError("No valid results collected from any fold")

    # Create aggregate visualizations across all folds
    create_cross_fold_visualizations(fold_results, output_base_dir)

    print(f"\nEvaluation complete! Results saved to: {output_base_dir}")
    return fold_results


def create_cross_fold_visualizations(fold_results, output_dir):
    """Create visualizations aggregating results across all folds"""
    # Implementation for cross-fold visualization will go here
    pass


def get_latest_model_dir(fold_dir, protein_name, norm_type):
    """Find the most recent model directory in a fold using full protein name"""
    models_dir = fold_dir / "models"
    print(f"\nDebug - Looking in directory: {models_dir}")

    if not models_dir.exists():
        raise FileNotFoundError(f"No models directory found in {fold_dir}")

    # Print the contents of models directory
    print("Debug - Contents of models directory:")
    model_dirs = []

    # Clean protein name
    cleaned_protein_name = protein_name.strip()
    print(f"Debug - Cleaned protein name: '{cleaned_protein_name}'")

    for item in models_dir.iterdir():
        print(f"Debug - Checking directory: '{item.name}'")
        # Changed from startswith to in
        print(f"Debug - Does dir contain protein name? {cleaned_protein_name in item.name}")
        print(f"Debug - Does dir contain norm type? {norm_type in item.name}")

        # Changed from startswith to in
        if cleaned_protein_name in item.name and norm_type in item.name:
            print(f"Debug - Match found!")
            model_dirs.append(item)

    if not model_dirs:
        raise FileNotFoundError(
            f"No model directories found for protein '{cleaned_protein_name}' and {norm_type} in {models_dir}")

    return max(model_dirs, key=lambda x: x.stat().st_mtime)


# Define global paths
THESIS_DIR = Path(r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\Thesis")
DATASET_DIR = THESIS_DIR / "dataset"

if __name__ == "__main__":
    # Configuration
    # PROTEIN_NAME = " Ubiquitin-conjugating enzyme E2 L3"
    # PROTEIN_NAME = " Elongation factor 1-alpha 1"
    # PROTEIN_NAME = " Actin, cytoplasmic 2"
    # PROTEIN_NAME = " Annexin A7"
    # PROTEIN_NAME = " Thioredoxin"
    # PROTEIN_NAME = " Gelsolin"
    # PROTEIN_NAME = " 14-3-3 protein beta_alpha"
    # PROTEIN_NAME = " Cornulin"
    PROTEIN_NAME = " Proliferation marker protein Ki-67"
    # NORM_TYPE = "Intensity"
    NORM_TYPE = "LFQ"
    N_FOLDS = 5
    N_RUNS = 20
    N_TILES_PER_SLIDE = 100

    try:
        fold_results = evaluate_protein_model_cv_multi_run(
            protein_name=PROTEIN_NAME,
            norm_type=NORM_TYPE,
            n_folds=N_FOLDS,
            n_runs=N_RUNS,
            n_tiles_per_slide=N_TILES_PER_SLIDE
        )
        print("\nMulti-run evaluation complete!")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
