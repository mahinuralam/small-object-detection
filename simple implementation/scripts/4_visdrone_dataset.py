"""
Step 4: VisDrone Dataset for Faster R-CNN
Handles loading and conversion to Faster R-CNN format
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple

class VisDroneDataset(Dataset):
    """
    VisDrone dataset for Faster R-CNN
    
    VisDrone format: (x, y, w, h) with (x,y) = top-left corner in pixels
    Faster R-CNN expects: (x1, y1, x2, y2) in pixels
    """
    
    # VisDrone class mapping (1-10 are valid, 0 and 11 are ignored)
    CLASS_NAMES = [
        'ignored',      # 0 - ignored regions
        'pedestrian',   # 1
        'people',       # 2
        'bicycle',      # 3
        'car',          # 4
        'van',          # 5
        'truck',        # 6
        'tricycle',     # 7
        'awning-tricycle', # 8
        'bus',          # 9
        'motor',        # 10
        'others'        # 11 - ignored
    ]
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transforms=None,
        min_size: int = 5
    ):
        """
        Args:
            root_dir: Path to VisDrone-2018 directory
            split: 'train', 'val', or 'test'
            transforms: Optional transforms (albumentations or torchvision)
            min_size: Minimum box width/height to keep (filter tiny boxes)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms
        self.min_size = min_size
        
        # Find the split directory
        self.split_dir = self._find_split_dir()
        if self.split_dir is None:
            raise ValueError(f"Could not find split '{split}' in {root_dir}")
        
        # Get image and annotation paths
        self.img_dir = self.split_dir / 'images'
        self.ann_dir = self.split_dir / 'annotations'
        
        if not self.img_dir.exists() or not self.ann_dir.exists():
            raise ValueError(f"Images or annotations not found in {self.split_dir}")
        
        # Get all image files
        self.image_files = sorted(list(self.img_dir.glob('*.jpg')))
        
        print(f"Loaded {len(self.image_files)} images from {split} split")
        
    def _find_split_dir(self):
        """Find the directory for the requested split"""
        for split_dir in self.root_dir.iterdir():
            if not split_dir.is_dir():
                continue
            
            split_name = split_dir.name.lower()
            
            # Check if this is the right split
            if self.split == 'train' and 'train' in split_name:
                # Check structure
                img_dir = split_dir / 'images' if (split_dir / 'images').exists() else split_dir / split_dir.name / 'images'
                if img_dir.exists():
                    return split_dir if (split_dir / 'images').exists() else split_dir / split_dir.name
            
            elif self.split == 'val' and 'val' in split_name:
                img_dir = split_dir / 'images' if (split_dir / 'images').exists() else split_dir / split_dir.name / 'images'
                if img_dir.exists():
                    return split_dir if (split_dir / 'images').exists() else split_dir / split_dir.name
            
            elif self.split == 'test' and 'test' in split_name:
                img_dir = split_dir / 'images' if (split_dir / 'images').exists() else split_dir / split_dir.name / 'images'
                if img_dir.exists():
                    return split_dir if (split_dir / 'images').exists() else split_dir / split_dir.name
        
        return None
    
    def __len__(self):
        return len(self.image_files)
    
    def _load_annotation(self, ann_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load annotation file and convert to Faster R-CNN format
        
        Returns:
            boxes: (N, 4) array of [x1, y1, x2, y2] in pixels
            labels: (N,) array of class labels (1-10)
        """
        boxes = []
        labels = []
        
        if not ann_path.exists():
            # Return empty arrays
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                
                # Parse annotation
                x, y, w, h = map(float, parts[:4])
                score = int(parts[4])
                cls_id = int(parts[5])
                
                # Filter invalid annotations
                if score == 0:  # Ignored by score
                    continue
                if cls_id == 0 or cls_id == 11:  # Ignored classes
                    continue
                if w < self.min_size or h < self.min_size:  # Too small
                    continue
                
                # Convert to corner format: (x1, y1, x2, y2)
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id)  # Keep original class IDs (1-10)
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        return boxes, labels
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get image and target dictionary for Faster R-CNN
        
        Returns:
            image: (3, H, W) tensor
            target: dict with 'boxes', 'labels', 'image_id', 'area', 'iscrowd'
        """
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load annotations
        ann_path = self.ann_dir / f"{img_path.stem}.txt"
        boxes, labels = self._load_annotation(ann_path)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Calculate area
        if len(boxes) > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.zeros((0,), dtype=torch.float32)
        
        # Create target dictionary (Faster R-CNN format)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # Apply transforms if provided
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            # Default: PIL → tensor via raw bytes (no numpy C-ext dependency)
            w, h = image.size
            image = torch.ByteTensor(bytearray(image.tobytes())).reshape(h, w, 3).permute(2, 0, 1).float() / 255.0
        
        return image, target
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for batching
        Images can have different sizes, so we return lists
        """
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets


def test_dataset():
    """Test the dataset loader"""
    print("="*80)
    print("TESTING VISDRONE DATASET LOADER")
    print("="*80)
    
    # Load validation dataset
    dataset = VisDroneDataset(
        root_dir='../dataset/VisDrone-2018',
        split='val',
        min_size=5
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Class names: {dataset.CLASS_NAMES[1:11]}")  # Valid classes
    
    # Test loading a few samples
    print("\nTesting sample loading...")
    for i in range(min(3, len(dataset))):
        image, target = dataset[i]
        
        print(f"\nSample {i+1}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Num objects: {len(target['boxes'])}")
        
        if len(target['boxes']) > 0:
            print(f"  Boxes shape: {target['boxes'].shape}")
            print(f"  Labels shape: {target['labels'].shape}")
            print(f"  Sample box: {target['boxes'][0]}")
            print(f"  Sample label: {target['labels'][0].item()} ({dataset.CLASS_NAMES[target['labels'][0].item()]})")
            
            # Verify box format (x1 < x2, y1 < y2)
            boxes = target['boxes']
            valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            print(f"  Valid boxes: {valid_boxes.sum().item()}/{len(boxes)}")
    
    # Test dataloader
    print("\n" + "-"*80)
    print("Testing DataLoader...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=VisDroneDataset.collate_fn,
        num_workers=0
    )
    
    batch_images, batch_targets = next(iter(loader))
    print(f"\nBatch size: {len(batch_images)}")
    for i, (img, target) in enumerate(zip(batch_images, batch_targets)):
        print(f"  Sample {i+1}: image shape {img.shape}, {len(target['boxes'])} objects")
    
    print("\n" + "="*80)
    print("Dataset loader test PASSED ✓")
    print("="*80)


if __name__ == '__main__':
    test_dataset()
