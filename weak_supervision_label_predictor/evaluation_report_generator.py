import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
from protein_expression_model import TileDataset, ProteinExpressionModel
from torch.utils.data import DataLoader

# Define global paths
THESIS_DIR = Path(r"C:\Users\eliran.shmi\Documents\Thesis")
DATASET_DIR = THESIS_DIR / "dataset"
TILES_DATASET_DIR = DATASET_DIR / "tiles_dataset_100x100"


class ProteinExpressionEvaluator:
    def __init__(self, model_dir):
        """Initialize evaluator with a trained model"""
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.load_model()
        print(f"Loaded model from {model_dir}")

    def load_model(self):
        """Load the trained model and configuration"""
        # Load training configuration
        config_path = self.model_dir / "training_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load model checkpoint
        model_path = self.model_dir / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path)
        self.model = ProteinExpressionModel(model_type=checkpoint['model_type'])
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.model.eval()

        print("\nModel configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

    def evaluate(self, test_loader, output_dir=None):
        """Evaluate model performance"""
        print("\nStarting model evaluation...")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Results will be saved to: {output_dir}")

        # Collect predictions
        predictions = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating tiles"):
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

        # Convert to DataFrame
        results_df = pd.DataFrame(predictions)
        print(f"\nProcessed {len(results_df)} tiles")

        # Calculate metrics at different levels
        slide_metrics = self._calculate_slide_metrics(results_df)
        print(f"Aggregated metrics for {len(slide_metrics)} slides")

        patient_metrics = self._calculate_patient_metrics(results_df)
        print(f"Aggregated metrics for {len(patient_metrics)} patients")

        # Calculate correlations
        correlation_metrics = self._calculate_correlations(slide_metrics)
        print("\nCorrelation metrics:")
        for pred_type, metrics in correlation_metrics.items():
            print(f"  {pred_type}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")

        # Create comprehensive evaluation report
        evaluation_report = self._create_evaluation_report(
            results_df, slide_metrics, patient_metrics, correlation_metrics
        )

        if output_dir:
            # Save results
            results_df.to_csv(output_dir / "tile_predictions.csv", index=False)
            slide_metrics.to_csv(output_dir / "slide_metrics.csv", index=False)
            patient_metrics.to_csv(output_dir / "patient_metrics.csv", index=False)

            # Save plots
            self._save_evaluation_plots(
                results_df, slide_metrics, patient_metrics, correlation_metrics, output_dir
            )

            # Save report
            with open(output_dir / "evaluation_report.json", 'w') as f:
                json.dump(evaluation_report, f, indent=4)
            print(f"\nSaved all results to {output_dir}")

        return evaluation_report

    def _calculate_slide_metrics(self, results_df):
        """Calculate metrics at the slide level"""
        print("\nCalculating slide-level metrics...")

        slide_metrics = []
        for slide_id in tqdm(results_df['slide_id'].unique(), desc="Processing slides"):
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

    def _calculate_patient_metrics(self, results_df):
        """Calculate metrics at the patient level"""
        print("\nCalculating patient-level metrics...")

        patient_metrics = []
        for patient_id in results_df['patient_id'].unique():
            patient_data = results_df[results_df['patient_id'] == patient_id]
            patient_slides = patient_data['slide_id'].unique()

            # Get ground truth info (same for all slides of the patient)
            true_label = patient_data['true_label'].iloc[0]
            expression_value = patient_data['expression_value'].iloc[0]

            # Calculate metrics across all tiles
            n_tiles = len(patient_data)
            n_slides = len(patient_slides)
            n_positive = sum(patient_data['tile_prediction'] > 0.5)
            positive_ratio = n_positive / n_tiles
            mean_prediction = patient_data['tile_prediction'].mean()
            std_prediction = patient_data['tile_prediction'].std()

            patient_metrics.append({
                'patient_id': patient_id,
                'true_label': true_label,
                'expression_value': expression_value,
                'n_slides': n_slides,
                'n_tiles': n_tiles,
                'n_positive_tiles': n_positive,
                'positive_ratio': positive_ratio,
                'mean_prediction': mean_prediction,
                'std_prediction': std_prediction,
                'predicted_label': positive_ratio > 0.5
            })

        return pd.DataFrame(patient_metrics)

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
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            }

        return correlations

    def _create_evaluation_report(self, results_df, slide_metrics, patient_metrics,
                                  correlation_metrics):
        """Create comprehensive evaluation report"""
        print("\nCreating evaluation report...")

        # Calculate accuracies at different levels
        tile_predictions = results_df['tile_prediction'] > 0.5
        tile_labels = results_df['true_label']
        tile_accuracy = np.mean(tile_predictions == tile_labels)

        slide_predictions = slide_metrics['predicted_label']
        slide_labels = slide_metrics['true_label']
        slide_accuracy = np.mean(slide_predictions == slide_labels)

        patient_predictions = patient_metrics['predicted_label']
        patient_labels = patient_metrics['true_label']
        patient_accuracy = np.mean(patient_predictions == patient_labels)

        report = {
            'model_config': self.config,
            'dataset_stats': {
                'n_tiles': len(results_df),
                'n_slides': len(slide_metrics),
                'n_patients': len(patient_metrics)
            },
            'tile_level_metrics': {
                'accuracy': float(tile_accuracy)
            },
            'slide_level_metrics': {
                'accuracy': float(slide_accuracy),
                'mean_tiles_per_slide': float(slide_metrics['n_tiles'].mean()),
                'std_tiles_per_slide': float(slide_metrics['n_tiles'].std())
            },
            'patient_level_metrics': {
                'accuracy': float(patient_accuracy),
                'mean_slides_per_patient': float(patient_metrics['n_slides'].mean()),
                'std_slides_per_patient': float(patient_metrics['n_slides'].std())
            },
            'correlation_metrics': correlation_metrics,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return report

    def _save_evaluation_plots(self, results_df, slide_metrics, patient_metrics,
                               correlation_metrics, output_dir):
        """Create and save evaluation plots"""
        print("\nCreating evaluation plots...")

        # Use standard style
        plt.style.use('default')

        # 1. Tile prediction distribution
        plt.figure(figsize=(12, 6))
        plt.hist([
            results_df[results_df['true_label'] == 0]['tile_prediction'],
            results_df[results_df['true_label'] == 1]['tile_prediction']
        ], label=['Negative', 'Positive'], bins=50, alpha=0.7)
        plt.title('Distribution of Tile Predictions by True Label')
        plt.xlabel('Prediction Score')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(output_dir / "tile_predictions_dist.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Expression correlation plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Positive ratio correlation
        ax1.scatter(slide_metrics['expression_value'], slide_metrics['positive_ratio'])
        ax1.set_title(f'Expression vs Positive Tiles Ratio\n' +
                      f'r={correlation_metrics["positive_ratio"]["pearson_r"]:.3f}, ' +
                      f'ρ={correlation_metrics["positive_ratio"]["spearman_r"]:.3f}')
        ax1.set_xlabel('Expression Value')
        ax1.set_ylabel('Ratio of Positive Tiles')

        # Mean prediction correlation
        ax2.scatter(slide_metrics['expression_value'], slide_metrics['mean_prediction'])
        ax2.set_title(f'Expression vs Mean Prediction\n' +
                      f'r={correlation_metrics["mean_prediction"]["pearson_r"]:.3f}, ' +
                      f'ρ={correlation_metrics["mean_prediction"]["spearman_r"]:.3f}')
        ax2.set_xlabel('Expression Value')
        ax2.set_ylabel('Mean Prediction')

        plt.tight_layout()
        plt.savefig(output_dir / "correlation_plots.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Prediction distribution by level (tile/slide/patient)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Tile-level predictions
        neg_data = slide_metrics[slide_metrics['true_label'] == 0]['positive_ratio']
        pos_data = slide_metrics[slide_metrics['true_label'] == 1]['positive_ratio']

        bp1 = ax1.boxplot([neg_data, pos_data],
                          positions=[0, 1],
                          labels=['Negative', 'Positive'])
        ax1.set_title('Tile-Level Predictions')
        ax1.set_ylabel('Ratio of Positive Predictions')

        # Slide-level predictions
        neg_data = slide_metrics[slide_metrics['true_label'] == 0]['mean_prediction']
        pos_data = slide_metrics[slide_metrics['true_label'] == 1]['mean_prediction']

        bp2 = ax2.boxplot([neg_data, pos_data],
                          positions=[0, 1],
                          labels=['Negative', 'Positive'])
        ax2.set_title('Slide-Level Predictions')
        ax2.set_ylabel('Mean Prediction')

        # Patient-level predictions
        neg_data = patient_metrics[patient_metrics['true_label'] == 0]['mean_prediction']
        pos_data = patient_metrics[patient_metrics['true_label'] == 1]['mean_prediction']

        bp3 = ax3.boxplot([neg_data, pos_data],
                          positions=[0, 1],
                          labels=['Negative', 'Positive'])
        ax3.set_title('Patient-Level Predictions')
        ax3.set_ylabel('Mean Prediction')

        plt.tight_layout()
        plt.savefig(output_dir / "prediction_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()


def get_latest_model_dir(fold_dir):
    """Find the most recent model directory in a fold"""
    models_dir = fold_dir / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"No models directory found in {fold_dir}")

    model_dirs = list(models_dir.glob("*_fold*_*"))
    if not model_dirs:
        raise FileNotFoundError(f"No model directories found in {models_dir}")

    # Sort by creation time and get most recent
    return max(model_dirs, key=lambda x: x.stat().st_mtime)


def calculate_binary_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall, and F1 score"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'support': {
            'total': int(tp + tn + fp + fn),
            'positive': int(tp + fn),
            'negative': int(tn + fp)
        }
    }


def evaluate_protein_model_cv(protein_name, norm_type='Intensity', n_folds=5, batch_size=32):
    """Evaluate model across all CV folds and aggregate results"""
    try:
        print(f"\nEvaluating CV models for protein: {protein_name}")

        # Get dataset directory
        dataset_dir = DATASET_DIR / f"tiles_dataset_100x100_{protein_name}_{norm_type}_cv"
        if not dataset_dir.exists():
            raise FileNotFoundError(f"CV Dataset directory not found: {dataset_dir}")

        # Lists to collect metrics for each fold
        all_fold_metrics = {
            'tile': [],
            'slide': [],
            'patient': []
        }
        all_slide_metrics = []
        model_timestamps = []

        for fold in range(n_folds):
            try:
                print(f"\nEvaluating fold {fold + 1}/{n_folds}")

                # Get fold-specific directories
                fold_dir = dataset_dir / f"fold_{fold}"
                model_dir = get_latest_model_dir(fold_dir)
                model_timestamps.append(model_dir.name.split('_')[-1])

                print(f"Using model: {model_dir.name}")

                # Load evaluator and evaluate
                evaluator = ProteinExpressionEvaluator(model_dir)
                eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fold_output_dir = model_dir / f"evaluation_{eval_timestamp}"
                fold_output_dir.mkdir(parents=True, exist_ok=True)

                # Get test data and evaluate
                labels_df = pd.read_csv(fold_dir / "dataset_info.csv")
                test_df = labels_df[labels_df['split'] == 'test'].copy()

                if len(test_df) == 0:
                    raise ValueError(f"No test set data found for fold {fold}")

                test_dataset = TileDataset(fold_dir, test_df, split='test')
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

                # Evaluate fold
                fold_result = evaluator.evaluate(test_loader, fold_output_dir)

                # Process tile-level metrics
                tile_preds = np.array([p['prediction'] for p in fold_result['tile_level_metrics']])
                tile_labels = np.array([p['true_label'] for p in fold_result['tile_level_metrics']])

                # Process slide-level metrics
                slide_metrics = fold_result['slide_level_metrics']
                slide_preds = np.array([p['mean_prediction'] for p in slide_metrics])
                slide_labels = np.array([p['true_label'] for p in slide_metrics])
                expression_values = np.array([p['expression_value'] for p in slide_metrics])
                positive_ratios = np.array([p['positive_ratio'] for p in slide_metrics])

                # Process patient-level metrics
                patient_preds = np.array([p['mean_prediction'] for p in fold_result['patient_level_metrics']])
                patient_labels = np.array([p['true_label'] for p in fold_result['patient_level_metrics']])

                # Calculate binary metrics for each level
                binary_metrics = {
                    'tile': calculate_binary_metrics(tile_labels, tile_preds > 0.5),
                    'slide': calculate_binary_metrics(slide_labels, slide_preds > 0.5),
                    'patient': calculate_binary_metrics(patient_labels, patient_preds > 0.5)
                }

                # Add fold information
                for level in binary_metrics:
                    binary_metrics[level]['fold'] = fold
                    all_fold_metrics[level].append(binary_metrics[level])

                # Create and collect slide metrics DataFrame
                slide_data = pd.DataFrame({
                    'true_label': slide_labels,
                    'mean_prediction': slide_preds,
                    'expression_value': expression_values,
                    'positive_ratio': positive_ratios,
                    'fold': fold,
                    'slide_id': [p['slide_id'] for p in slide_metrics],
                    'patient_id': [p['patient_id'] for p in slide_metrics]
                })
                all_slide_metrics.append(slide_data)

            except Exception as e:
                print(f"Error processing fold {fold}: {str(e)}")
                continue

        if not all_slide_metrics:
            raise ValueError("No valid results collected from any fold")

        # Create aggregate output directory
        eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        aggregate_output_dir = dataset_dir / f"aggregate_evaluation_{eval_timestamp}"
        aggregate_output_dir.mkdir(parents=True, exist_ok=True)

        # Combine slide metrics and calculate correlations
        combined_slide_metrics = pd.concat(all_slide_metrics, ignore_index=True)

        # Calculate correlations with p-values
        correlations = {}
        for pred_col in ['positive_ratio', 'mean_prediction']:
            pearson_r, pearson_p = stats.pearsonr(
                combined_slide_metrics['expression_value'],
                combined_slide_metrics[pred_col]
            )
            spearman_r, spearman_p = stats.spearmanr(
                combined_slide_metrics['expression_value'],
                combined_slide_metrics[pred_col]
            )

            correlations[pred_col] = {
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p)
            }

        # Create aggregate plots with p-values
        create_aggregate_plots(
            combined_slide_metrics,
            correlations,
            aggregate_output_dir
        )

        # Calculate aggregate metrics
        aggregate_metrics = {}
        for level in ['tile', 'slide', 'patient']:
            metrics_df = pd.DataFrame(all_fold_metrics[level])

            aggregate_metrics[level] = {
                'mean': {
                    'accuracy': float(metrics_df['accuracy'].mean()),
                    'precision': float(metrics_df['precision'].mean()),
                    'recall': float(metrics_df['recall'].mean()),
                    'f1_score': float(metrics_df['f1_score'].mean())
                },
                'std': {
                    'accuracy': float(metrics_df['accuracy'].std()),
                    'precision': float(metrics_df['precision'].std()),
                    'recall': float(metrics_df['recall'].std()),
                    'f1_score': float(metrics_df['f1_score'].std())
                },
                'per_fold': all_fold_metrics[level],
                'support': {
                    'total': sum(m['support']['total'] for m in all_fold_metrics[level]),
                    'positive': sum(m['support']['positive'] for m in all_fold_metrics[level]),
                    'negative': sum(m['support']['negative'] for m in all_fold_metrics[level])
                }
            }

        # Save aggregate results
        combined_slide_metrics.to_csv(aggregate_output_dir / "all_slide_metrics.csv", index=False)
        with open(aggregate_output_dir / "aggregate_evaluation.json", 'w') as f:
            json.dump({
                'protein_name': protein_name,
                'norm_type': norm_type,
                'n_folds': n_folds,
                'model_timestamps': model_timestamps,
                'metrics': aggregate_metrics,
                'correlations': correlations
            }, f, indent=4)

        # Print detailed results
        print("\nAggregate Results:")
        for level in ['tile', 'slide', 'patient']:
            print(f"\n{level.capitalize()}-level metrics:")
            metrics = aggregate_metrics[level]['mean']
            std = aggregate_metrics[level]['std']
            support = aggregate_metrics[level]['support']
            print(
                f"  Total samples: {support['total']} (Positive: {support['positive']}, Negative: {support['negative']})")
            print(f"  Accuracy:  {metrics['accuracy']:.3f} ± {std['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f} ± {std['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f} ± {std['recall']:.3f}")
            print(f"  F1 Score:  {metrics['f1_score']:.3f} ± {std['f1_score']:.3f}")

        print("\nCorrelations:")
        for pred_type in ['positive_ratio', 'mean_prediction']:
            print(f"\n{pred_type}:")
            corr = correlations[pred_type]
            print(f"  Pearson r:  {corr['pearson_r']:.3f} (p={corr['pearson_p']:.3e})")
            print(f"  Spearman ρ: {corr['spearman_r']:.3f} (p={corr['spearman_p']:.3e})")

        return aggregate_metrics, correlations

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise


def calculate_aggregate_correlations(slide_metrics_df):
    """Calculate correlations using all test slides across folds"""
    correlations = {}

    for pred_col in ['positive_ratio', 'mean_prediction']:
        pearson_r, pearson_p = stats.pearsonr(
            slide_metrics_df[pred_col],
            slide_metrics_df['expression_value']
        )

        spearman_r, spearman_p = stats.spearmanr(
            slide_metrics_df[pred_col],
            slide_metrics_df['expression_value']
        )

        correlations[pred_col] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        }

    return correlations


def create_aggregate_plots(slide_metrics_df, correlations, output_dir):
    """Create plots using aggregated data from all folds"""

    # 1. Correlation plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Expression vs Positive Tiles Ratio
    ratio_corr = correlations['positive_ratio']
    ax1.scatter(slide_metrics_df['expression_value'], slide_metrics_df['positive_ratio'])
    ax1.set_title('Expression vs Positive Tiles Ratio\n' +
                  f'r={ratio_corr["pearson_r"]:.3f} (p={ratio_corr["pearson_p"]:.1e})\n' +
                  f'ρ={ratio_corr["spearman_r"]:.3f} (p={ratio_corr["spearman_p"]:.1e})')
    ax1.set_xlabel('Expression Value')
    ax1.set_ylabel('Ratio of Positive Tiles')

    # Expression vs Mean Prediction
    mean_corr = correlations['mean_prediction']
    ax2.scatter(slide_metrics_df['expression_value'], slide_metrics_df['mean_prediction'])
    ax2.set_title('Expression vs Mean Prediction\n' +
                  f'r={mean_corr["pearson_r"]:.3f} (p={mean_corr["pearson_p"]:.1e})\n' +
                  f'ρ={mean_corr["spearman_r"]:.3f} (p={mean_corr["spearman_p"]:.1e})')
    ax2.set_xlabel('Expression Value')
    ax2.set_ylabel('Mean Prediction')

    plt.tight_layout()
    plt.savefig(output_dir / "aggregate_correlation_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Distribution of Tile Predictions by True Label
    plt.figure(figsize=(12, 6))
    neg_preds = slide_metrics_df[slide_metrics_df['true_label'] == 0]['mean_prediction']
    pos_preds = slide_metrics_df[slide_metrics_df['true_label'] == 1]['mean_prediction']

    plt.hist([neg_preds, pos_preds], label=['Negative', 'Positive'],
             bins=30, alpha=0.7, density=True)
    plt.title('Distribution of Predictions by True Label (All Folds)')
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(output_dir / "aggregate_prediction_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Prediction Distributions by Level (Boxplots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Prediction Distributions by Level (All Folds)')

    # Tile-Level
    neg_data = slide_metrics_df[slide_metrics_df['true_label'] == 0]['positive_ratio']
    pos_data = slide_metrics_df[slide_metrics_df['true_label'] == 1]['positive_ratio']

    axes[0].boxplot([neg_data, pos_data], tick_labels=['Negative', 'Positive'])
    axes[0].set_title('Tile-Level Predictions')
    axes[0].set_ylabel('Ratio of Positive Predictions')

    # Slide-Level
    neg_data = slide_metrics_df[slide_metrics_df['true_label'] == 0]['mean_prediction']
    pos_data = slide_metrics_df[slide_metrics_df['true_label'] == 1]['mean_prediction']

    axes[1].boxplot([neg_data, pos_data], tick_labels=['Negative', 'Positive'])
    axes[1].set_title('Slide-Level Predictions')
    axes[1].set_ylabel('Mean Prediction')

    # Patient-Level
    patient_metrics = slide_metrics_df.groupby('patient_id').agg({
        'true_label': 'first',
        'mean_prediction': 'mean'
    })
    neg_data = patient_metrics[patient_metrics['true_label'] == 0]['mean_prediction']
    pos_data = patient_metrics[patient_metrics['true_label'] == 1]['mean_prediction']

    axes[2].boxplot([neg_data, pos_data], tick_labels=['Negative', 'Positive'])
    axes[2].set_title('Patient-Level Predictions')
    axes[2].set_ylabel('Mean Prediction')

    plt.tight_layout()
    plt.savefig(output_dir / "aggregate_prediction_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Optional: Expression Value Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(slide_metrics_df['expression_value'], bins=30, alpha=0.7)
    plt.title('Distribution of Expression Values (All Folds)')
    plt.xlabel('Expression Value')
    plt.ylabel('Count')
    plt.savefig(output_dir / "aggregate_expression_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()


def aggregate_existing_evaluations(protein_name, norm_type='Intensity', n_folds=5):
    """Aggregate existing evaluation results from all folds without recalculating"""
    try:
        print(f"\nAggregating existing evaluations for protein: {protein_name}")

        # Get dataset directory
        dataset_dir = DATASET_DIR / f"tiles_dataset_100x100_{protein_name}_{norm_type}_cv"
        if not dataset_dir.exists():
            raise FileNotFoundError(f"CV Dataset directory not found: {dataset_dir}")

        # Collect metrics from each fold
        all_fold_metrics = {
            'tile': [],
            'slide': [],
            'patient': []
        }
        all_slide_metrics = []
        model_timestamps = []

        for fold in range(n_folds):
            try:
                print(f"\nProcessing fold {fold + 1}/{n_folds}")

                # Get fold directory and latest evaluation
                fold_dir = dataset_dir / f"fold_{fold}"
                model_dir = get_latest_model_dir(fold_dir)
                eval_dirs = list(model_dir.glob("evaluation_*"))
                if not eval_dirs:
                    raise ValueError(f"No evaluation directory found for fold {fold}")
                latest_eval_dir = max(eval_dirs, key=lambda x: x.stat().st_mtime)

                # Record model timestamp
                model_timestamps.append(model_dir.name.split('_')[-1])

                print(f"Using evaluation results from: {latest_eval_dir}")

                # Read metrics from saved files
                slide_metrics = pd.read_csv(latest_eval_dir / "slide_metrics.csv")
                slide_metrics['fold'] = fold
                all_slide_metrics.append(slide_metrics)

                # Calculate fold-specific metrics
                fold_metrics = get_fold_metrics(slide_metrics)

                # Add fold information
                for level in fold_metrics:
                    fold_metrics[level]['fold'] = fold
                    all_fold_metrics[level].append(fold_metrics[level])

            except Exception as e:
                print(f"Error processing fold {fold}: {str(e)}")
                continue

        if not all_slide_metrics:
            raise ValueError("No valid results collected from any fold")

        # Create aggregate output directory
        eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        aggregate_output_dir = dataset_dir / f"aggregate_evaluation_{eval_timestamp}"
        aggregate_output_dir.mkdir(parents=True, exist_ok=True)

        # Combine slide metrics and calculate overall metrics
        combined_slide_metrics = pd.concat(all_slide_metrics, ignore_index=True)

        # Calculate overall metrics for all levels
        overall_metrics = get_fold_metrics(combined_slide_metrics)

        # Create detailed metrics report
        metrics_report = []
        metrics_report.append(f"Detailed Metrics Report for {protein_name}")
        metrics_report.append("=" * 50)
        metrics_report.append(f"\nNormalization Type: {norm_type}")
        metrics_report.append(f"Number of Folds: {n_folds}")
        metrics_report.append("\nPER-LEVEL METRICS")
        metrics_report.append("=" * 50)

        for level in ['tile', 'slide', 'patient']:
            metrics_report.append(f"\n{level.upper()}-LEVEL METRICS")
            metrics_report.append("=" * 30)
            metrics_report.append(format_metrics_table(overall_metrics[level]))

            # Add fold-specific metrics
            metrics_report.append("\nPer-Fold Performance:")
            metrics_report.append("-" * 20)
            for fold_metric in all_fold_metrics[level]:
                fold_num = fold_metric['fold']
                metrics_report.append(f"\nFold {fold_num}:")
                metrics_report.append(f"  Accuracy:  {fold_metric['accuracy']:.3f}")
                metrics_report.append(f"  Precision: {fold_metric['precision']:.3f}")
                metrics_report.append(f"  Recall:    {fold_metric['recall']:.3f}")
                metrics_report.append(f"  F1 Score:  {fold_metric['f1_score']:.3f}")

        # Calculate correlations
        correlations = calculate_aggregate_correlations(combined_slide_metrics)

        # Add correlation results to report
        metrics_report.append("\nCORRELATION ANALYSIS")
        metrics_report.append("=" * 30)
        for pred_type in ['positive_ratio', 'mean_prediction']:
            metrics_report.append(f"\n{pred_type}:")
            corr = correlations[pred_type]
            metrics_report.append(f"  Pearson r:  {corr['pearson_r']:.3f} (p={corr['pearson_p']:.3e})")
            metrics_report.append(f"  Spearman ρ: {corr['spearman_r']:.3f} (p={corr['spearman_p']:.3e})")

        # Save detailed metrics report
        with open(aggregate_output_dir / "detailed_metrics_report.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(metrics_report))

        # Create aggregate plots
        create_aggregate_plots(
            combined_slide_metrics,
            correlations,
            aggregate_output_dir
        )

        # Prepare aggregate results
        aggregate_results = {
            'protein_name': protein_name,
            'norm_type': norm_type,
            'n_folds': n_folds,
            'model_timestamps': model_timestamps,
            'overall_metrics': overall_metrics,
            'per_fold_metrics': {
                level: all_fold_metrics[level] for level in ['tile', 'slide', 'patient']
            },
            'correlations': correlations
        }

        # Save aggregate results
        combined_slide_metrics.to_csv(aggregate_output_dir / "all_slide_metrics.csv", index=False)
        with open(aggregate_output_dir / "aggregate_results.json", 'w') as f:
            json.dump(aggregate_results, f, indent=4)

        # Print summary results
        print("\nAggregate Results Summary:")
        for level in ['tile', 'slide', 'patient']:
            print(f"\n{level.capitalize()}-level metrics:")
            metrics = overall_metrics[level]
            print(
                f"  Total samples: {metrics['support']['total']} (Pos: {metrics['support']['positive']}, Neg: {metrics['support']['negative']})")
            print(f"  Accuracy:  {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1 Score:  {metrics['f1_score']:.3f}")

        print("\nCorrelations:")
        for pred_type in ['positive_ratio', 'mean_prediction']:
            print(f"\n{pred_type}:")
            corr = correlations[pred_type]
            print(f"  Pearson r:  {corr['pearson_r']:.3f} (p={corr['pearson_p']:.3e})")
            print(f"  Spearman ρ: {corr['spearman_r']:.3f} (p={corr['spearman_p']:.3e})")

        print(f"\nDetailed results saved to: {aggregate_output_dir}")
        return aggregate_results

    except Exception as e:
        print(f"Error during aggregation: {str(e)}")
        raise


def calculate_all_metrics(y_true, y_pred_probs, threshold=0.5):
    """Calculate all binary classification metrics"""
    y_pred = y_pred_probs > threshold
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'support': {
            'total': int(tp + tn + fp + fn),
            'positive': int(tp + fn),
            'negative': int(tn + fp)
        },
        'confusion_matrix': {
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    }


def get_fold_metrics(slide_metrics_df):
    """Calculate metrics at all levels for a single fold"""
    # Slide-level metrics
    slide_metrics = calculate_all_metrics(
        slide_metrics_df['true_label'].values,
        slide_metrics_df['mean_prediction'].values
    )

    # Patient-level metrics
    patient_df = slide_metrics_df.groupby('patient_id').agg({
        'true_label': 'first',
        'mean_prediction': 'mean'
    })
    patient_metrics = calculate_all_metrics(
        patient_df['true_label'].values,
        patient_df['mean_prediction'].values
    )

    # Tile-level metrics (using positive_ratio as proxy for tile predictions)
    tile_metrics = calculate_all_metrics(
        slide_metrics_df['true_label'].values,
        slide_metrics_df['positive_ratio'].values
    )

    return {
        'tile': tile_metrics,
        'slide': slide_metrics,
        'patient': patient_metrics
    }


def format_metrics_table(metrics):
    """Create a formatted metrics table as string"""
    lines = []
    lines.append("Classification Metrics:")
    lines.append("--------------------")
    lines.append(f"Accuracy:  {metrics['accuracy']:.3f}")
    lines.append(f"Precision: {metrics['precision']:.3f}")
    lines.append(f"Recall:    {metrics['recall']:.3f}")
    lines.append(f"F1 Score:  {metrics['f1_score']:.3f}")
    lines.append("\nSupport:")
    lines.append(f"Total samples:     {metrics['support']['total']}")
    lines.append(f"Positive samples:  {metrics['support']['positive']}")
    lines.append(f"Negative samples:  {metrics['support']['negative']}")
    lines.append("\nConfusion Matrix:")
    lines.append(f"True Positives:  {metrics['confusion_matrix']['tp']}")
    lines.append(f"True Negatives:  {metrics['confusion_matrix']['tn']}")
    lines.append(f"False Positives: {metrics['confusion_matrix']['fp']}")
    lines.append(f"False Negatives: {metrics['confusion_matrix']['fn']}")
    return "\n".join(lines)


if __name__ == "__main__":
    PROTEIN_NAME = " Thioredoxin"
    # PROTEIN_NAME = " Ubiquitin-conjugating enzyme E2 L3; Ubiquitin-conjugating enzyme E2 L5"
    # PROTEIN_NAME = " Annexin A7"
    NORM_TYPE = "Intensity"
    # NORM_TYPE = "LFQ"
    TIMESTAMP = "20250123_091022"  # Replace with actual timestamp

    dataset_base_dir = DATASET_DIR / f"tiles_dataset_100x100_{PROTEIN_NAME}_{NORM_TYPE}_cv"

    # try:
    #     aggregate_results = evaluate_protein_model_cv(
    #         protein_name=PROTEIN_NAME,
    #         norm_type=NORM_TYPE,
    #         n_folds=5,
    #         batch_size=32
    #     )
    #     print("\nEvaluation complete!")
    # except Exception as e:
    #     print(f"Error during evaluation: {str(e)}")

    try:
        aggregate_results = aggregate_existing_evaluations(
            protein_name=PROTEIN_NAME,
            norm_type=NORM_TYPE,
            n_folds=5
        )
        print("\nAggregation complete!")
    except Exception as e:
        print(f"Error during aggregation: {str(e)}")
