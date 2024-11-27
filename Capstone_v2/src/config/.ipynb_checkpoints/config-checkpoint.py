# src/config/config.py

class Config:
    """Configuration class to store all parameters"""
    def __init__(self):
        # Dataset configuration
        self.USE_DATASET_FRACTION = True
        self.DATASET_FRACTION = 0.5
        self.USE_DYNAMIC_SAMPLING = False
        
        # Image configuration
        self.IMG_SIZE = (256, 256)
        self.IMG_SHAPE = self.IMG_SIZE + (3,)
        
        # Training configuration
        self.BATCH_SIZE = 32
        self.EPOCH_COUNT = 15
        self.USE_AUGMENTATION = True
        
        # Path configuration
        self.SAVE_DIR = "best_models"
        self.LOG_CSV = "model_metrics.csv"
        self.BASE_PATH = "/Users/keith/iCloud/Imperial/Module 25/data/real_and_fake_imageset"
        
        # Model configuration
        self.OPTIMIZATION_STRATEGY = "None"  # Options: "None", "BO", "Optuna"
        self.DATASETS = ["Data Set 1/Data Set 1"]
        
        # Active models configuration
        self.ACTIVE_MODELS = {
            "CNN": False,
            "Mesonet": True,
            "SVM": False,
            "Random Forest": False
        }
