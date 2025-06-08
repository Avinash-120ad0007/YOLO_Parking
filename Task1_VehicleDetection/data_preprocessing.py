# import os
# import yaml
# import shutil
# import logging
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# import cv2
# import numpy as np
# from typing import List, Tuple

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('preprocessing.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# class ParkingDataPreprocessor:
#     """
#     Handles all data preprocessing tasks for parking space detection
#     """
    
#     def __init__(self, data_root: str, output_dir: str = "datasets/parking_processed"):
#         self.data_root = Path(data_root)
#         self.output_dir = Path(output_dir)
#         self.classes = ['empty_space', 'occupied_space']  # Standard parking classes
        
#         # Create output directories
#         self.setup_directories()
        
#     def setup_directories(self):
#         """Create necessary directory structure"""
#         logger.info("Setting up directory structure...")
        
#         for split in ['train', 'val', 'test']:
#             for folder in ['images', 'labels']:
#                 (self.output_dir / split / folder).mkdir(parents=True, exist_ok=True)
        
#         logger.info(f"Created directories at: {self.output_dir}")
    
#     def create_yaml_config(self):
#         """Create dataset configuration file for YOLO training"""
#         config = {
#             'path': str(self.output_dir.absolute()),
#             'train': 'train/images',
#             'val': 'val/images',
#             'test': 'test/images',
#             'nc': len(self.classes),
#             'names': self.classes
#         }
        
#         yaml_path = self.output_dir / 'data.yaml'
#         with open(yaml_path, 'w') as f:
#             yaml.dump(config, f, default_flow_style=False)
        
#         logger.info(f"Created YOLO config at: {yaml_path}")
#         return yaml_path
    
#     def process_roboflow_dataset(self, dataset_path: str, split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1)):
#         """
#         Process downloaded Roboflow dataset
#         Args:
#             dataset_path: Path to downloaded dataset
#             split_ratios: (train, val, test) ratios
#         """
#         logger.info(f"Processing Roboflow dataset from: {dataset_path}")
        
#         dataset_path = Path(dataset_path)
        
#         # Find all images and labels
#         image_files = []
#         for ext in ['*.jpg', '*.jpeg', '*.png']:
#             image_files.extend(list(dataset_path.rglob(ext)))
        
#         logger.info(f"Found {len(image_files)} images")
        
#         # Split dataset
#         train_files, temp_files = train_test_split(image_files, train_size=split_ratios[0], random_state=42)
#         val_files, test_files = train_test_split(temp_files, train_size=split_ratios[1]/(split_ratios[1]+split_ratios[2]), random_state=42)
        
#         # Process each split
#         splits = {
#             'train': train_files,
#             'val': val_files,
#             'test': test_files
#         }
        
#         for split_name, files in splits.items():
#             logger.info(f"Processing {split_name} split: {len(files)} files")
#             self._copy_files_to_split(files, split_name)
        
#         # Create YOLO config
#         self.create_yaml_config()
#         logger.info("Dataset preprocessing completed successfully!")
    
#     def _copy_files_to_split(self, image_files: List[Path], split_name: str):
#         """Copy image and label files to appropriate split directory"""
#         for img_path in image_files:
#             # Copy image
#             dst_img = self.output_dir / split_name / 'images' / img_path.name
#             shutil.copy2(img_path, dst_img)
            
#             # Find and copy corresponding label file
#             label_path = img_path.parent / 'labels' / f"{img_path.stem}.txt"
#             if not label_path.exists():
#                 # Try different label directory structures
#                 possible_label_paths = [
#                     img_path.with_suffix('.txt'),
#                     img_path.parent.parent / 'labels' / f"{img_path.stem}.txt",
#                 ]
#                 for possible_path in possible_label_paths:
#                     if possible_path.exists():
#                         label_path = possible_path
#                         break
            
#             if label_path.exists():
#                 dst_label = self.output_dir / split_name / 'labels' / f"{img_path.stem}.txt"
#                 shutil.copy2(label_path, dst_label)
#             else:
#                 logger.warning(f"No label file found for {img_path.name}")
    
#     def validate_dataset(self):
#         """Validate the processed dataset"""
#         logger.info("Validating processed dataset...")
        
#         for split in ['train', 'val', 'test']:
#             img_dir = self.output_dir / split / 'images'
#             label_dir = self.output_dir / split / 'labels'
            
#             img_count = len(list(img_dir.glob('*')))
#             label_count = len(list(label_dir.glob('*')))
            
#             logger.info(f"{split.upper()}: {img_count} images, {label_count} labels")
            
#             if img_count != label_count:
#                 logger.warning(f"Mismatch in {split}: {img_count} images vs {label_count} labels")
        
#         logger.info("Dataset validation completed!")
    
#     def augment_data(self, augment_factor: int = 2):
#         """
#         Apply data augmentation to training set
#         Args:
#             augment_factor: Number of augmented versions per original image
#         """
#         logger.info(f"Applying data augmentation with factor: {augment_factor}")
        
#         train_img_dir = self.output_dir / 'train' / 'images'
#         train_label_dir = self.output_dir / 'train' / 'labels'
        
#         original_images = list(train_img_dir.glob('*'))
        
#         for img_path in original_images:
#             img = cv2.imread(str(img_path))
#             if img is None:
#                 continue
                
#             label_path = train_label_dir / f"{img_path.stem}.txt"
#             if not label_path.exists():
#                 continue
            
#             # Read labels
#             with open(label_path, 'r') as f:
#                 labels = f.read().strip()
            
#             for i in range(augment_factor):
#                 # Apply random augmentations
#                 aug_img = self._apply_augmentation(img)
                
#                 # Save augmented image and labels
#                 aug_img_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"
#                 aug_label_name = f"{img_path.stem}_aug_{i}.txt"
                
#                 cv2.imwrite(str(train_img_dir / aug_img_name), aug_img)
                
#                 with open(train_label_dir / aug_label_name, 'w') as f:
#                     f.write(labels)
        
#         logger.info("Data augmentation completed!")
    
#     def _apply_augmentation(self, img: np.ndarray) -> np.ndarray:
#         """Apply random augmentations to image"""
#         # Random brightness
#         if np.random.random() > 0.5:
#             brightness = np.random.uniform(0.7, 1.3)
#             img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
        
#         # Random horizontal flip
#         if np.random.random() > 0.5:
#             img = cv2.flip(img, 1)
        
#         # Random noise
#         if np.random.random() > 0.5:
#             noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
#             img = cv2.add(img, noise)
        
#         return img

# def main():
#     """Main preprocessing function"""
#     logger.info("Starting data preprocessing for parking space detection...")
    
#     # Initialize preprocessor
#     preprocessor = ParkingDataPreprocessor(
#         data_root="data/raw",
#         output_dir="datasets/parking_processed"
#     )
    
#     # Download instructions
#     logger.info("""
#     DATASET DOWNLOAD INSTRUCTIONS:
#     1. Visit: https://universe.roboflow.com/browse/logistics/parking
#     2. Choose a suitable parking dataset (recommended: PKLot or similar)
#     3. Download in YOLO format
#     4. Extract to 'data/raw/' directory
#     5. Run this script again
#     """)
    
#     # Check if raw data exists
#     raw_data_path = Path("data/raw")
#     if raw_data_path.exists() and any(raw_data_path.iterdir()):
#         preprocessor.process_roboflow_dataset("data/raw")
#         preprocessor.validate_dataset()
#         preprocessor.augment_data(augment_factor=1)  # Light augmentation
#     else:
#         logger.warning("No raw data found. Please download dataset first.")
#         return False
    
#     return True

# if __name__ == "__main__":
#     success = main()
#     if success:
#         logger.info("Data preprocessing completed successfully!")
#     else:
#         logger.error("Data preprocessing failed!")

import yaml
import logging
from pathlib import Path
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MinimalPKLotSetup:
    """
    Minimal setup for PKLot dataset - just creates config and validates
    """
    
    def __init__(self, dataset_path: str = "PKLot.v2-640.yolov12"):
        self.dataset_path = Path(dataset_path)
        
    def create_yaml_config(self) -> Path:
        """Create YOLO configuration file for the existing dataset"""
        
        # First, let's inspect the dataset to get class information
        classes = self.get_dataset_classes()
        
        config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'train/images',
            'val': 'valid/images',  # Keep original 'valid' naming
            'test': 'test/images',
            'nc': len(classes),
            'names': classes
        }
        
        yaml_path = self.dataset_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Created YOLO config at: {yaml_path}")
        return yaml_path
    
    def get_dataset_classes(self) -> list:
        """Analyze label files to determine class names and count"""
        logger.info("Analyzing dataset classes...")
        
        # Check if existing data.yaml has class information
        existing_yaml = self.dataset_path / 'data.yaml'
        if existing_yaml.exists():
            try:
                with open(existing_yaml, 'r') as f:
                    existing_config = yaml.safe_load(f)
                    if 'names' in existing_config:
                        logger.info(f"Found existing classes: {existing_config['names']}")
                        return existing_config['names']
            except Exception as e:
                logger.warning(f"Could not read existing data.yaml: {e}")
        
        # Analyze label files to determine classes
        class_ids = set()
        label_dirs = [
            self.dataset_path / 'train' / 'labels',
            self.dataset_path / 'valid' / 'labels',
            self.dataset_path / 'test' / 'labels'
        ]
        
        for label_dir in label_dirs:
            if label_dir.exists():
                for label_file in label_dir.glob('*.txt'):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    class_id = int(line.strip().split()[0])
                                    class_ids.add(class_id)
                    except (ValueError, IndexError):
                        continue
        
        # Create class names based on found IDs
        if class_ids:
            max_class = max(class_ids)
            # Common parking space classes
            class_names = []
            for i in range(max_class + 1):
                if i == 0:
                    class_names.append('empty_space')
                elif i == 1:
                    class_names.append('occupied_space')
                else:
                    class_names.append(f'class_{i}')
            
            logger.info(f"Detected {len(class_names)} classes: {class_names}")
            return class_names
        else:
            logger.warning("No class information found, using default parking classes")
            return ['empty_space', 'occupied_space']
    
    def validate_dataset(self):
        """Validate the existing dataset structure"""
        logger.info("Validating PKLot dataset structure...")
        
        splits = ['train', 'valid', 'test']
        total_images = 0
        total_labels = 0
        
        for split in splits:
            img_dir = self.dataset_path / split / 'images'
            label_dir = self.dataset_path / split / 'labels'
            
            if img_dir.exists() and label_dir.exists():
                images = list(img_dir.glob('*.[jJ][pP][gG]')) + list(img_dir.glob('*.[pP][nN][gG]'))
                labels = list(label_dir.glob('*.txt'))
                
                logger.info(f"{split.upper()}: {len(images)} images, {len(labels)} labels")
                total_images += len(images)
                total_labels += len(labels)
                
                # Check for missing labels
                missing_labels = []
                for img_file in images:
                    label_file = label_dir / f"{img_file.stem}.txt"
                    if not label_file.exists():
                        missing_labels.append(img_file.name)
                
                if missing_labels:
                    logger.warning(f"{split}: {len(missing_labels)} images missing labels")
                    if len(missing_labels) <= 5:
                        logger.warning(f"Missing labels for: {missing_labels}")
            else:
                logger.error(f"{split} directory structure is incomplete")
        
        logger.info(f"Total dataset: {total_images} images, {total_labels} labels")
        return total_images > 0 and total_labels > 0
    
    def get_dataset_stats(self):
        """Get comprehensive dataset statistics"""
        logger.info("Generating dataset statistics...")
        
        stats = {}
        class_distribution = Counter()
        
        splits = ['train', 'valid', 'test']
        
        for split in splits:
            label_dir = self.dataset_path / split / 'labels'
            if not label_dir.exists():
                continue
                
            split_stats = {'images': 0, 'annotations': 0, 'classes': Counter()}
            
            for label_file in label_dir.glob('*.txt'):
                split_stats['images'] += 1
                
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                class_id = int(line.strip().split()[0])
                                split_stats['annotations'] += 1
                                split_stats['classes'][class_id] += 1
                                class_distribution[class_id] += 1
                            except (ValueError, IndexError):
                                continue
            
            stats[split] = split_stats
            logger.info(f"{split.upper()}: {split_stats['images']} images, {split_stats['annotations']} annotations")
        
        # Overall class distribution
        logger.info("Overall class distribution:")
        for class_id, count in sorted(class_distribution.items()):
            percentage = (count / sum(class_distribution.values())) * 100
            logger.info(f"  Class {class_id}: {count} annotations ({percentage:.1f}%)")
        
        return stats
    
    def setup_dataset(self):
        """Complete minimal setup process"""
        logger.info("Setting up PKLot dataset for YOLO training...")
        
        # Validate dataset exists and has proper structure
        if not self.dataset_path.exists():
            logger.error(f"Dataset path {self.dataset_path} does not exist!")
            return False
        
        # Validate dataset structure
        if not self.validate_dataset():
            logger.error("Dataset validation failed!")
            return False
        
        # Create YAML configuration
        yaml_path = self.create_yaml_config()
        
        # Generate statistics
        self.get_dataset_stats()
        
        logger.info("✅ Dataset setup completed!")
        logger.info(f"✅ Configuration saved to: {yaml_path}")
        logger.info("✅ Dataset is ready for YOLO training!")
        
        return True

def main():
    """Main function"""
    setup = MinimalPKLotSetup(r"/teamspace/studios/this_studio/Task1_VehicleDetection/data")
    
    success = setup.setup_dataset()
    
    if success:
        logger.info("""
        🎯 Next Steps:
        1. Your dataset is ready for training!
        2. Use the generated data.yaml file for YOLO training
        3. No additional preprocessing needed
        4. Start training with: yolo train data=PKLot.v2-640.yolov12/data.yaml model=yolov8n.pt
        """)
    else:
        logger.error("Setup failed! Please check your dataset structure.")
    
    return success

if __name__ == "__main__":
    main()