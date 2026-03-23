import os
from pathlib import Path
import torch
import torch.nn as nn
from torch import GradScaler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Define global paths
THESIS_DIR = Path(r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\Thesis")
TIFFS_DIR = THESIS_DIR / "2021-01-17" / "Tiffs"
CSVS_DIR = THESIS_DIR / "CSVs"
DATASET_DIR = THESIS_DIR / "dataset"


# TILES_DATASET_DIR = DATASET_DIR / "tiles_dataset_100x100"


class TileDataset(Dataset):
    """Dataset for WSI tiles with patient-level labels"""

    def __init__(self, split_dir, labels_df, split, transform=None):
        """
        Initialize dataset

        Args:
            split_dir: Base directory containing splits (train/validation/test)
            labels_df: DataFrame with slide labels
            split: Which split to use ('train', 'validation', or 'test')
            transform: Optional transforms to apply to images
        """
        self.split_dir = Path(split_dir) / split
        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")

        self.labels_df = labels_df[labels_df['split'] == split].copy()

        # Create list of all tiles and their labels
        self.tiles = []
        for _, row in self.labels_df.iterrows():
            slide_dir = self.split_dir / row['slide_id']
            if not slide_dir.exists():
                print(f"Warning: Directory not found for slide {row['slide_id']}")
                continue

            slide_tiles = list(slide_dir.glob('*.jpg'))
            print(f"Found {len(slide_tiles)} tiles for slide {row['slide_id']}")

            for tile_path in slide_tiles:
                self.tiles.append({
                    'path': tile_path,
                    'label': row['label'],
                    'slide_id': row['slide_id'],
                    'patient_id': row['patient_id'],
                    'expression_value': row['expression_value']
                })

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"{split} dataset created with {len(self.tiles)} tiles from {len(self.labels_df)} slides")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile_info = self.tiles[idx]
        try:
            image = Image.open(tile_info['path'])
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {tile_info['path']}: {str(e)}")
            # Return a blank image in case of error
            image = torch.zeros((3, 224, 224))

        return {
            'image': image,
            'label': torch.tensor(tile_info['label'], dtype=torch.float32),
            'slide_id': tile_info['slide_id'],
            'patient_id': tile_info['patient_id'],
            'expression_value': tile_info['expression_value']
        }


class ProteinExpressionModel:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', model_type='resnet18'):
        self.device = device
        self.model_type = model_type
        self.model = self._create_model()
        self.model.to(device)
        print(f"Model created on device: {device}")

    def _create_model(self):
        print(f"Creating {self.model_type} model...")
        if self.model_type == 'resnet18':
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        return model

    def train_model(self, train_loader, val_loader,
                    num_epochs=10, learning_rate=0.001,
                    save_dir=None):
        """Train the model with validation"""
        print("\nStarting model training...")
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scaler = GradScaler()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=2, verbose=True
        )

        best_val_loss = float('inf')
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_tile_accuracy': [],
            'val_patient_level_accuracy': [],
            'lr': []
        }

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 20)

            # Training phase
            train_loss = self._train_epoch(train_loader, criterion, optimizer, scaler)

            # Validation phase
            val_loss, val_metrics = self._validate(val_loader, criterion)

            # Update learning rate
            scheduler.step(val_loss)

            # Save history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['val_tile_accuracy'].append(val_metrics['tile_accuracy'])
            training_history['val_patient_level_accuracy'].append(val_metrics['patient_accuracy'])
            training_history['lr'].append(optimizer.param_groups[0]['lr'])

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Tile Accuracy: {val_metrics['tile_accuracy']:.4f}")
            print(f"Val Patient-Level Accuracy: {val_metrics['patient_accuracy']:.4f}")

            # Save best model
            if save_dir and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = Path(save_dir) / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'model_type': self.model_type
                }, save_path)
                print(f"Saved best model to {save_path}")

        # Save training history
        if save_dir:
            self._save_training_history(training_history, save_dir)

        return training_history

    def _train_epoch(self, train_loader, criterion, optimizer, scaler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")

        for batch in progress_bar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # Update weights
            scaler.update()  # Update the scaler

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / len(train_loader)

    def _validate(self, val_loader, criterion):
        """Perform validation"""
        self.model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        patient_predictions = {}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Store predictions by patient
                for i in range(len(outputs)):
                    patient_id = batch['patient_id'][i]
                    if patient_id not in patient_predictions:
                        patient_predictions[patient_id] = {
                            'preds': [],
                            'true': batch['label'][i].item()
                        }
                    patient_predictions[patient_id]['preds'].append(outputs[i].item())

                predictions.extend(outputs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Calculate tile-level accuracy
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        tile_accuracy = np.mean((predictions > 0.5) == true_labels)

        # Calculate patient-level accuracy
        patient_correct = 0
        for patient_data in patient_predictions.values():
            mean_pred = np.mean(patient_data['preds'])
            patient_pred = mean_pred > 0.5
            patient_correct += patient_pred == patient_data['true']

        patient_accuracy = patient_correct / len(patient_predictions)

        return (val_loss / len(val_loader),
                {'tile_accuracy': tile_accuracy,
                 'patient_accuracy': patient_accuracy})

    def _save_training_history(self, history, save_dir):
        """Save training history plots and data"""
        save_dir = Path(save_dir)

        # Save history data
        history_df = pd.DataFrame(history)
        history_df.to_csv(save_dir / "training_history.csv", index=False)

        # Create and save plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Accuracy plot
        ax2.plot(history['val_tile_accuracy'], label='Tile Accuracy')
        ax2.plot(history['val_patient_level_accuracy'], label='Patient Accuracy')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(history['lr'], label='Learning Rate', color='r', linestyle='--')
        ax2.set_title('Validation Metrics')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2_twin.set_ylabel('Learning Rate')

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2)

        plt.tight_layout()
        plt.savefig(save_dir / "training_history.png")
        plt.close()


def train_protein_model_cv(protein_name, norm_type='Intensity',
                           num_epochs=10, batch_size=32, learning_rate=0.001, n_folds=5):
    """Train model using k-fold cross validation"""
    # First find the dataset directory with original path
    original_dataset_dir = DATASET_DIR / f"tiles_dataset_100x100_{protein_name}_{norm_type}_cv"
    print(f"Looking for dataset directory: {original_dataset_dir}")

    if not original_dataset_dir.exists():
        raise FileNotFoundError(f"CV Dataset directory not found: {original_dataset_dir}")

    print(f"\nTraining model for protein: {protein_name} using {n_folds}-fold CV")
    print(f"Normalization type: {norm_type}")
    print(f"Using dataset directory: {original_dataset_dir}")

    cv_results = []
    for fold in range(n_folds):
        print(f"\nProcessing fold {fold + 1}/{n_folds}")
        fold_dir = original_dataset_dir / f"fold_{fold}"
        print(f"Fold directory: {fold_dir}")

        # Load fold-specific data
        labels_df = pd.read_csv(fold_dir / "dataset_info.csv")
        print(f"Loaded labels for fold {fold}")

        # Create models directory first
        models_dir = fold_dir / "models"
        print(f"Creating models directory: {models_dir}")
        if not models_dir.exists():
            try:
                models_dir.mkdir(exist_ok=True)
                print(f"Created models directory: {models_dir}")
            except Exception as e:
                print(f"Failed to create models directory: {str(e)}")
                continue

        # Create specific model directory using short name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{protein_name}_{norm_type}_f{fold}_{timestamp}"
        output_dir = models_dir / model_name

        try:
            print(f"Creating model directory: {output_dir}")
            output_dir.mkdir(exist_ok=True)
            print(f"Created model directory: {output_dir}")
        except Exception as e:
            print(f"Failed to create model directory: {str(e)}")
            continue

        try:
            # Create datasets
            train_dataset = TileDataset(fold_dir, labels_df, split='train')
            val_dataset = TileDataset(fold_dir, labels_df, split='validation')

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            # Initialize and train model
            model = ProteinExpressionModel()
            history = model.train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                save_dir=output_dir
            )

            # Save fold configuration
            config = {
                'fold': fold,
                'protein_name': protein_name,
                'norm_type': norm_type,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'model_type': model.model_type,
                'timestamp': timestamp,
                'train_tiles': len(train_dataset),
                'val_tiles': len(val_dataset),
                'train_slides': len(labels_df[labels_df['split'] == 'train']),
                'val_slides': len(labels_df[labels_df['split'] == 'validation']),
                'train_patients': len(labels_df[labels_df['split'] == 'train']['patient_id'].unique()),
                'val_patients': len(labels_df[labels_df['split'] == 'validation']['patient_id'].unique())
            }

            with open(output_dir / "training_config.json", 'w') as f:
                json.dump(config, f, indent=4)

            cv_results.append({
                'fold': fold,
                'output_dir': output_dir,
                'history': history,
                'config': config
            })

        except Exception as e:
            print(f"Error during training: {str(e)}")
            print(f"All directories in fold directory:")
            try:
                print([x.name for x in fold_dir.iterdir()])
            except Exception as e2:
                print(f"Error listing directory: {str(e2)}")
            continue

    return cv_results


if __name__ == "__main__":
    # Example usage
    # PROTEIN_NAME = " Thioredoxin"  # Replace with actual protein name
    # PROTEIN_NAMES = [" Ubiquitin-conjugating enzyme E2 L3; Ubiquitin-conjugating enzyme E2 L5", " Annexin A7"]
    # PROTEIN_NAMES = [" Thioredoxin"]
    # PROTEIN_NAMES = [" Ubiquitin-conjugating enzyme E2 L3; Ubiquitin-conjugating enzyme E2 L5"]
    NORM_TYPES = ["LFQ", "Intensity"]
    # NORM_TYPES = ["LFQ"]
    # NORM_TYPES = ["Intensity"]

    # PROTEIN_NAMES = [" Ubiquitin-conjugating enzyme E2 L3; Ubiquitin-conjugating enzyme E2 L5"]
    # PROTEIN_NAMES = [" Actin, cytoplasmic 2"]
    # PROTEIN_NAMES = [" Annexin A7"]
    # PROTEIN_NAMES = [" Gelsolin"]
    # PROTEIN_NAMES = [" 14-3-3 protein beta_alpha"]
    # PROTEIN_NAMES = [" Cornulin"]
    PROTEIN_NAMES = [" Proliferation marker protein Ki-67"]

    for PROTEIN_NAME in PROTEIN_NAMES:
        for NORM_TYPE in NORM_TYPES:
            try:
                cv_results = train_protein_model_cv(
                    protein_name=PROTEIN_NAME,
                    norm_type=NORM_TYPE,
                    num_epochs=20,
                    n_folds=5
                )
                print("\nCross-validation training complete!")

            except Exception as e:
                print(f"Error during training: {str(e)}")
