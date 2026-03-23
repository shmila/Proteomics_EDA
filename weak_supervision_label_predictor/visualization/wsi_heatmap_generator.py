from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from weak_supervision_label_predictor.model.protein_expression_model import ProteinExpressionModel
import json


class SingleWSIDataset(Dataset):
    """Dataset for loading all tiles from a single WSI"""

    def __init__(self, tile_paths):
        self.tile_paths = tile_paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        try:
            image = Image.open(tile_path)
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {tile_path}: {str(e)}")
            image = torch.zeros((3, 224, 224))
        return {'image': image}


class WSIHeatmapGenerator:
    def __init__(self, base_dir, wsi_id, protein_name, norm_type='Intensity'):
        """
        Initialize heatmap generator for a specific WSI and protein.

        Args:
            base_dir: Base directory containing project data
            wsi_id: ID of the WSI (e.g., '1M02')
            protein_name: Name of the protein to analyze
            norm_type: Type of normalization ('Intensity', 'iBAQ', or 'LFQ')
        """
        self.base_dir = Path(base_dir)
        self.wsi_id = wsi_id
        self.protein_name = protein_name
        self.norm_type = norm_type

        # Set up directories
        self.dataset_dir = self.base_dir / "dataset" / f"tiles_dataset_100x100_{protein_name}_{norm_type}_cv"
        self.wsi_dir = self.base_dir / "2021-01-17" / "Tiffs" / wsi_id
        self.output_dir = self.base_dir / "heatmaps" / f"{protein_name}_{norm_type}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nInitializing heatmap generator:")
        print(f"WSI: {wsi_id}")
        print(f"Protein: {protein_name}")
        print(f"Normalization: {norm_type}")

        # Load model
        self.model = self._load_best_model()

    def _load_best_model(self):
        """Load the best performing model from CV folds"""
        best_model = None
        best_val_loss = float('inf')

        # Check each fold for the best model
        for fold_dir in self.dataset_dir.glob("fold_*"):
            models_dir = fold_dir / "models"
            if not models_dir.exists():
                continue

            # Find any model directories
            print(f"Looking for models in: {models_dir}")
            model_dirs = list(models_dir.glob("*"))  # Get all directories

            if not model_dirs:
                print(f"No model directories found in {models_dir}")
                continue

            print(f"Found model directories: {[d.name for d in model_dirs]}")

            latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
            checkpoint_path = latest_model_dir / "best_model.pth"

            if not checkpoint_path.exists():
                continue

            # Load checkpoint and check validation loss
            checkpoint = torch.load(checkpoint_path)
            val_loss = checkpoint.get('val_loss', float('inf'))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model = ProteinExpressionModel()
                model.model.load_state_dict(checkpoint['model_state_dict'])
                best_model = model
                print(f"Found better model in {fold_dir.name} (val_loss: {val_loss:.4f})")

        if best_model is None:
            raise ValueError(f"No valid model found for {self.protein_name}")

        return best_model

    def _collect_tile_paths(self):
        """Collect all tile paths and their grid positions"""
        tile_dirs = list(self.wsi_dir.glob(f"*tiles_100x100 good tiles"))
        if not tile_dirs:
            raise ValueError(f"No tile directory found for WSI {self.wsi_id}")

        tile_dir = tile_dirs[0]
        tile_info = []

        for tile_path in tile_dir.glob("*.jpg"):
            # Extract position from filename
            # Format: {wsi_id}_tile_{i}_{j}.jpg
            parts = tile_path.stem.split('_')
            i, j = int(parts[-2]), int(parts[-1])

            tile_info.append({
                'path': tile_path,
                'row': i,
                'col': j
            })

        return pd.DataFrame(tile_info)

    def generate_heatmap(self):
        """Generate and save expression heatmap for the WSI"""
        print("\nGenerating heatmap...")

        # Collect tile information
        tiles_df = self._collect_tile_paths()
        print(f"Found {len(tiles_df)} tiles")

        # Create dataset
        dataset = SingleWSIDataset(tiles_df['path'].values)

        # Run inference
        self.model.model.eval()
        predictions = []

        with torch.no_grad():
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            for batch in tqdm(dataloader, desc="Running inference"):
                outputs = self.model.model(batch['image'].to(self.model.device)).squeeze()
                predictions.extend(outputs.cpu().numpy())

        # Add predictions to DataFrame
        tiles_df['prediction'] = predictions

        # Create heatmap matrix
        grid_size = 100
        heatmap = np.full((grid_size, grid_size), np.nan)

        for _, tile in tiles_df.iterrows():
            heatmap[tile['row'], tile['col']] = tile['prediction']

        # Plot heatmap
        plt.figure(figsize=(12, 12))
        mask = np.isnan(heatmap)
        sns.heatmap(heatmap,
                    mask=mask,
                    cmap='RdYlBu_r',
                    center=0.5,
                    square=True)

        plt.title(f"Protein Expression Heatmap\n{self.protein_name} ({self.wsi_id})")

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"heatmap_{self.wsi_id}_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Save prediction data
        tiles_df.to_csv(self.output_dir / f"predictions_{self.wsi_id}_{timestamp}.csv", index=False)

        print(f"\nHeatmap saved to: {output_path}")

        # Generate statistics
        stats = {
            'wsi_id': self.wsi_id,
            'protein_name': self.protein_name,
            'norm_type': self.norm_type,
            'n_tiles': len(tiles_df),
            'mean_prediction': float(tiles_df['prediction'].mean()),
            'std_prediction': float(tiles_df['prediction'].std()),
            'min_prediction': float(tiles_df['prediction'].min()),
            'max_prediction': float(tiles_df['prediction'].max()),
            'timestamp': timestamp
        }

        with open(self.output_dir / f"stats_{self.wsi_id}_{timestamp}.json", 'w') as f:
            json.dump(stats, f, indent=4)

        return output_path


def generate_wsi_heatmap(base_dir, wsi_id, protein_name, norm_type='Intensity'):
    """
    Generate protein expression heatmap for a specific WSI.

    Args:
        base_dir: Base directory containing project data
        wsi_id: ID of the WSI (e.g., '1M02')
        protein_name: Name of the protein to analyze
        norm_type: Type of normalization ('Intensity', 'iBAQ', or 'LFQ')
    """
    try:
        generator = WSIHeatmapGenerator(base_dir, wsi_id, protein_name, norm_type)
        output_path = generator.generate_heatmap()
        return output_path
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        raise


if __name__ == "__main__":
    # Configuration
    BASE_DIR = Path(r"C:\Users\elira\ShmilaJustSolveIt Dropbox\Eliran Shmila\PC\Documents\Thesis")
    WSI_ID = "1M02"
    # PROTEIN_NAME = " Thioredoxin"
    PROTEIN_NAME = " Elongation factor 1-alpha 1"
    # NORM_TYPE = "Intensity"
    NORM_TYPE = "LFQ"

    try:
        output_path = generate_wsi_heatmap(
            base_dir=BASE_DIR,
            wsi_id=WSI_ID,
            protein_name=PROTEIN_NAME,
            norm_type=NORM_TYPE
        )
        print(f"\nHeatmap generation complete!")
        print(f"Output saved to: {output_path}")
    except Exception as e:
        print(f"Error: {str(e)}")