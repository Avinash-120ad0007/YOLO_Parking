# import os
# import logging
# import torch
# from ultralytics import YOLO
# from pathlib import Path
# import yaml
# import matplotlib.pyplot as plt
# import time

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('training.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# class ParkingSpaceTrainer:
#     """
#     Handles training of YOLO model for parking space detection
#     """
    
#     def __init__(self, config_path: str, model_size: str = 'n'):
#         self.config_path = Path(config_path)
#         self.model_size = model_size  # n, s, m, l, x
#         self.model_name = f'yolov8{model_size}.pt'
        
#         # Training configuration
#         self.training_config = {
#             'epochs': 100,
#             'imgsz': 640,
#             'batch': 16,
#             'lr0': 0.01,
#             'weight_decay': 0.0005,
#             'patience': 20,
#             'save_period': 10,
#             'device': 'auto',  # Will use GPU if available
#             'workers': 4,
#             'project': 'runs/train',
#             'name': 'parking_detection',
#             'exist_ok': True
#         }
        
#         # Validation thresholds
#         self.min_map50 = 0.5  # Minimum mAP@0.5 for good model
        
#         logger.info(f"Initialized trainer with model: {self.model_name}")
#         logger.info(f"GPU Available: {torch.cuda.is_available()}")
#         if torch.cuda.is_available():
#             logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
#     def load_dataset_config(self):
#         """Load and validate dataset configuration"""
#         if not self.config_path.exists():
#             raise FileNotFoundError(f"Dataset config not found: {self.config_path}")
        
#         with open(self.config_path, 'r') as f:
#             config = yaml.safe_load(f)
        
#         logger.info(f"Dataset config loaded: {config}")
        
#         # Validate paths
#         for split in ['train', 'val']:
#             if split in config:
#                 split_path = Path(config['path']) / config[split]
#                 if not split_path.exists():
#                     raise FileNotFoundError(f"Dataset split not found: {split_path}")
        
#         return config
    
#     def initialize_model(self):
#         """Initialize YOLO model"""
#         logger.info(f"Initializing YOLOv8 model: {self.model_name}")
        
#         try:
#             # Load pre-trained model
#             model = YOLO(self.model_name)
#             logger.info("Model loaded successfully")
            
#             # Model info
#             model.info()
            
#             return model
#         except Exception as e:
#             logger.error(f"Failed to load model: {e}")
#             raise
    
#     def train_model(self, resume: bool = False):
#         """
#         Train the YOLO model
#         Args:
#             resume: Whether to resume training from last checkpoint
#         """
#         logger.info("Starting model training...")
        
#         # Load dataset config
#         dataset_config = self.load_dataset_config()
        
#         # Initialize model
#         model = self.initialize_model()
        
#         # Training arguments
#         train_args = {
#             'data': str(self.config_path),
#             'epochs': self.training_config['epochs'],
#             'imgsz': self.training_config['imgsz'],
#             'batch': self.training_config['batch'],
#             'lr0': self.training_config['lr0'],
#             'weight_decay': self.training_config['weight_decay'],
#             'patience': self.training_config['patience'],
#             'save_period': self.training_config['save_period'],
#             'device': self.training_config['device'],
#             'workers': self.training_config['workers'],
#             'project': self.training_config['project'],
#             'name': self.training_config['name'],
#             'exist_ok': self.training_config['exist_ok'],
#             'resume': resume,
#             'verbose': True
#         }
        
#         logger.info(f"Training arguments: {train_args}")
        
#         try:
#             # Start training
#             start_time = time.time()
#             results = model.train(**train_args)
            
#             training_time = time.time() - start_time
#             logger.info(f"Training completed in {training_time:.2f} seconds")
            
#             # Log training results
#             self.log_training_results(results)
            
#             return results
            
#         except Exception as e:
#             logger.error(f"Training failed: {e}")
#             raise
    
#     def log_training_results(self, results):
#         """Log comprehensive training results"""
#         logger.info("Training Results Summary:")
        
#         try:
#             # Extract key metrics
#             if hasattr(results, 'results_dict'):
#                 metrics = results.results_dict
                
#                 # Log key metrics
#                 for key, value in metrics.items():
#                     if isinstance(value, (int, float)):
#                         logger.info(f"  {key}: {value:.4f}")
            
#             # Model performance validation
#             best_fitness = getattr(results, 'best_fitness', 0)
#             logger.info(f"Best Fitness Score: {best_fitness:.4f}")
            
#             if best_fitness > self.min_map50:
#                 logger.info("Model achieved good performance!")
#             else:
#                 logger.warning(f"Model performance below threshold ({self.min_map50})")
            
#         except Exception as e:
#             logger.warning(f"Could not extract detailed metrics: {e}")
    
#     def validate_model(self, model_path: str = None):
#         """Validate trained model"""
#         if model_path is None:
#             model_path = f"runs/train/{self.training_config['name']}/weights/best.pt"
        
#         model_path = Path(model_path)
#         if not model_path.exists():
#             logger.error(f"Model weights not found: {model_path}")
#             return None
        
#         logger.info(f"Validating model: {model_path}")
        
#         try:
#             # Load trained model
#             model = YOLO(str(model_path))
            
#             # Run validation
#             val_results = model.val(
#                 data=str(self.config_path),
#                 imgsz=self.training_config['imgsz'],
#                 batch=self.training_config['batch'],
#                 device=self.training_config['device']
#             )
            
#             logger.info("Validation completed")
#             return val_results
            
#         except Exception as e:
#             logger.error(f"Validation failed: {e}")
#             return None
    
#     def create_training_plots(self):
#         """Create training visualization plots"""
#         results_dir = Path(f"runs/train/{self.training_config['name']}")
        
#         if not results_dir.exists():
#             logger.warning("Training results directory not found")
#             return
        
#         logger.info("Creating training plots...")
        
#         # Training plots are automatically generated by YOLO
#         plots_path = results_dir / "results.png"
#         if plots_path.exists():
#             logger.info(f"Training plots saved at: {plots_path}")
#         else:
#             logger.warning("Training plots not found")

# def main():
#     """Main training function"""
#     logger.info("Starting Parking Space Detection Training")
    
#     # Configuration
#     config_path = r"C:\Users\DELL\Desktop\Slabs\Task1_VehicleDetection\data\data.yaml"
#     model_size = 'n'  # Start with nano for speed, can upgrade to 's' or 'm'
    
#     # Check if dataset config exists
#     if not Path(config_path).exists():
#         logger.error(f"Dataset config not found: {config_path}")
#         logger.info("Please run data_preprocessing.py first!")
#         return False
    
#     try:
#         # Initialize trainer
#         trainer = ParkingSpaceTrainer(config_path, model_size)
        
#         # Train model
#         results = trainer.train_model()
        
#         # Validate model
#         val_results = trainer.validate_model()
        
#         # Create plots
#         trainer.create_training_plots()
        
#         logger.info("Training pipeline completed successfully!")
        
#         # Final model path
#         final_model_path = f"runs/train/{trainer.training_config['name']}/weights/best.pt"
#         logger.info(f"Best model saved at: {final_model_path}")
        
#         return True
        
#     except Exception as e:
#         logger.error(f"Training pipeline failed: {e}")
#         return False

# if __name__ == "__main__":
#     success = main()
#     if success:
#         logger.info("Training completed successfully!")
#     else:
#         logger.error("Training failed!")

import os
import logging
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import time

# Fix for PyTorch 2.6 compatibility
import torch.serialization
from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals([DetectionModel])

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ParkingSpaceTrainer:
    """
    Handles training of YOLO model for parking space detection
    """
    
    def __init__(self, config_path: str, model_size: str = 'n'):
        self.config_path = Path(config_path)
        self.model_size = model_size  # n, s, m, l, x
        self.model_name = f'yolov8{model_size}.pt'
        
        # Training configuration
        self.training_config = {
            'epochs': 10,
            'imgsz': 640,
            'batch': 8,
            'lr0': 0.01,
            'weight_decay': 0.0005,
            'patience': 20,
            'save_period': 10,
            'device': 'gpu',  # Will use GPU if available
            'workers': 4,
            'project': 'runs/train',
            'name': 'parking_detection',
            'exist_ok': True
        }
        
        # Validation thresholds
        self.min_map50 = 0.5  # Minimum mAP@0.5 for good model
        
        logger.info(f"Initialized trainer with model: {self.model_name}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    def load_dataset_config(self):
        """Load and validate dataset configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Dataset config loaded: {config}")
        
        # Validate paths
        for split in ['train', 'val']:
            if split in config:
                split_path = Path(config['path']) / config[split]
                if not split_path.exists():
                    raise FileNotFoundError(f"Dataset split not found: {split_path}")
        
        return config
    
    def initialize_model(self):
        """Initialize YOLO model with PyTorch 2.6 compatibility"""
        logger.info(f"Initializing YOLOv8 model: {self.model_name}")
        
        try:
            # Set torch.load to use weights_only=False for YOLO compatibility
            original_load = torch.load
            
            def patched_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            # Temporarily patch torch.load
            torch.load = patched_load
            
            # Load pre-trained model
            model = YOLO(self.model_name)
            
            # Restore original torch.load
            torch.load = original_load
            
            logger.info("Model loaded successfully")
            
            # Model info
            model.info()
            
            return model
            
        except Exception as e:
            # Restore original torch.load in case of error
            torch.load = original_load
            logger.error(f"Failed to load model: {e}")
            raise
    
    def train_model(self, resume: bool = False):
        """
        Train the YOLO model
        Args:
            resume: Whether to resume training from last checkpoint
        """
        logger.info("Starting model training...")
        
        # Load dataset config
        dataset_config = self.load_dataset_config()
        
        # Initialize model
        model = self.initialize_model()
        
        # Training arguments
        train_args = {
            'data': str(self.config_path),
            'epochs': self.training_config['epochs'],
            'imgsz': self.training_config['imgsz'],
            'batch': self.training_config['batch'],
            'lr0': self.training_config['lr0'],
            'weight_decay': self.training_config['weight_decay'],
            'patience': self.training_config['patience'],
            'save_period': self.training_config['save_period'],
            'device': self.training_config['device'],
            'workers': self.training_config['workers'],
            'project': self.training_config['project'],
            'name': self.training_config['name'],
            'exist_ok': self.training_config['exist_ok'],
            'resume': resume,
            'verbose': True
        }
        
        logger.info(f"Training arguments: {train_args}")
        
        try:
            # Start training
            start_time = time.time()
            results = model.train(**train_args)
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Log training results
            self.log_training_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def log_training_results(self, results):
        """Log comprehensive training results"""
        logger.info("Training Results Summary:")
        
        try:
            # Extract key metrics
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                
                # Log key metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {key}: {value:.4f}")
            
            # Model performance validation
            best_fitness = getattr(results, 'best_fitness', 0)
            logger.info(f"Best Fitness Score: {best_fitness:.4f}")
            
            if best_fitness > self.min_map50:
                logger.info("Model achieved good performance!")
            else:
                logger.warning(f"Model performance below threshold ({self.min_map50})")
            
        except Exception as e:
            logger.warning(f"Could not extract detailed metrics: {e}")
    
    def validate_model(self, model_path: str = None):
        """Validate trained model"""
        if model_path is None:
            model_path = f"runs/train/{self.training_config['name']}/weights/best.pt"
        
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"Model weights not found: {model_path}")
            return None
        
        logger.info(f"Validating model: {model_path}")
        
        try:
            # Load trained model with PyTorch 2.6 compatibility
            original_load = torch.load
            
            def patched_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            
            model = YOLO(str(model_path))
            
            torch.load = original_load
            
            # Run validation
            val_results = model.val(
                data=str(self.config_path),
                imgsz=self.training_config['imgsz'],
                batch=self.training_config['batch'],
                device=self.training_config['device']
            )
            
            logger.info("Validation completed")
            return val_results
            
        except Exception as e:
            torch.load = original_load
            logger.error(f"Validation failed: {e}")
            return None
    
    def create_training_plots(self):
        """Create training visualization plots"""
        results_dir = Path(f"runs/train/{self.training_config['name']}")
        
        if not results_dir.exists():
            logger.warning("Training results directory not found")
            return
        
        logger.info("Creating training plots...")
        
        # Training plots are automatically generated by YOLO
        plots_path = results_dir / "results.png"
        if plots_path.exists():
            logger.info(f"Training plots saved at: {plots_path}")
        else:
            logger.warning("Training plots not found")

def main():
    """Main training function"""
    logger.info("Starting Parking Space Detection Training")
    
    # Configuration
    config_path = r"/teamspace/studios/this_studio/Task1_VehicleDetection/data/data.yaml"
    model_size = 'n'  # Start with nano for speed, can upgrade to 's' or 'm'
    
    # Check if dataset config exists
    if not Path(config_path).exists():
        logger.error(f"Dataset config not found: {config_path}")
        logger.info("Please run data_preprocessing.py first!")
        return False
    
    try:
        # Initialize trainer
        trainer = ParkingSpaceTrainer(config_path, model_size)
        
        # Train model
        results = trainer.train_model()
        
        # Validate model
        val_results = trainer.validate_model()
        
        # Create plots
        trainer.create_training_plots()
        
        logger.info("Training pipeline completed successfully!")
        
        # Final model path
        final_model_path = f"runs/train/{trainer.training_config['name']}/weights/best.pt"
        logger.info(f"Best model saved at: {final_model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed!")