# data can be found at https://www.kaggle.com/datasets/shivamardeshna/real-and-fake-images-dataset-for-image-forensics/data

import os
from skopt.space import Real, Integer, Categorical

class Config:
    """Configuration class to store all parameters"""
    def __init__(self):
        # Path configuration
        self.BASE_PATH = "/Users/keith/Documents/CapstoneData/real_and_fake_imageset"
        self.SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_models")
        self.LOG_CSV = os.path.join(self.SAVE_DIR, "model_metrics.csv")
        
        # Dataset configuration
        self.USE_DATASET_FRACTION = True
        self.DATASET_FRACTION = 0.1
        self.USE_DYNAMIC_SAMPLING = False
        self.DATASETS = ["Data Set 1/Data Set 1"]
        
        # Image configuration
        self.IMG_SIZE = (128, 128)  # Updated for MesoNet
        self.IMG_SHAPE = self.IMG_SIZE + (3,)
        
        # Training configuration
        self.BATCH_SIZE = 32
        self.EPOCH_COUNT = 5
        self.USE_AUGMENTATION = True # image preprocessing toggle
        
        # Callback configuration
        self.REDUCE_LR_PATIENCE = 5
        self.REDUCE_LR_FACTOR = 0.5
        self.REDUCE_LR_MIN = 1e-7
        self.EARLY_STOPPING_PATIENCE = 10
        
        # Optimization configuration
        self.OPTIMIZATION_STRATEGY = "Optuna"  # Options: "None", "BO", "Optuna"
        self.OPTIMIZATION_TRIALS = 5
        self.OPTIMIZATION_RANDOM_STARTS = 5
        
        # Dashboard configuration
        self.SHOW_DASHBOARDS = True
        self.DASHBOARD_PORT = 8000
        
        # Data augmentation configuration
        self.AUGMENTATION_PARAMS = {
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'brightness_range': [0.8, 1.2],
            'horizontal_flip': True,
            'zoom_range': 0.2,
            'rescaling': 1./255
        }
        
        # Class weights for handling imbalanced data
        self.CLASS_WEIGHTS = {0: 1.5, 1: 1.0}
        
        # Model configurations
        self.MODEL_CONFIGS = {
            "CNN": {
                "default_params": {
                    "model_params": {
                        "learning_rate": 0.001,
                        "num_conv_layers": 3,
                        "num_filters": 32,
                        "kernel_size": (3, 3),
                        "pool_type": "max",
                        "dropout_rate": 0.5,
                        "batch_size": self.BATCH_SIZE
                    },
                    "training_params": {
                        "batch_size": self.BATCH_SIZE,
                        "epochs": self.EPOCH_COUNT,
                        "use_augmentation": self.USE_AUGMENTATION,
                        "class_weights": None
                    }
                },
                "search_space": [
                    Real(1e-4, 1e-2, name="learning_rate", prior="log-uniform"),
                    Integer(1, 4, name="num_conv_layers"),
                    Integer(32, 256, name="num_filters"),
                    Categorical(["3x3", "5x5"], name="kernel_size"),
                    Categorical(["max", "average"], name="pool_type"),
                    Real(0.1, 0.5, name="dropout_rate"),
                    Integer(16, 64, name="batch_size")
                ]
            },
            "Mesonet": {
                "default_params": {
                    "model_params": {
                        "initial_filters": 16,
                        "conv1_size": 3,
                        "conv_other_size": 5,
                        "dropout_rate": 0.5,
                        "dense_units": 32,
                        "leaky_relu_alpha": 0.1,
                        "learning_rate": 0.0001,
                        "batch_size": self.BATCH_SIZE
                    },
                    "training_params": {
                        "batch_size": self.BATCH_SIZE,
                        "epochs": self.EPOCH_COUNT,
                        "use_augmentation": self.USE_AUGMENTATION,
                        "class_weights": self.CLASS_WEIGHTS,
                        "augmentation_params": {
                            "rotation_range": 20,
                            "width_shift_range": 0.2,
                            "height_shift_range": 0.2,
                            "brightness_range": [0.8, 1.2],
                            "horizontal_flip": True,
                            "zoom_range": 0.2
                        }
                    }
                },
                "search_space": [
                    # Architecture parameters
                    Integer(8, 32, name="initial_filters"),
                    Integer(3, 5, name="conv1_size"),
                    Integer(3, 7, name="conv_other_size"),
                    Real(0.2, 0.7, name="dropout_rate"),
                    Integer(16, 64, name="dense_units"),
                    Real(0.01, 0.3, name="leaky_relu_alpha"),
                    
                    # Training parameters
                    Real(1e-5, 1e-3, name="learning_rate", prior="log-uniform"),
                    Categorical([16, 32, 64], name="batch_size"),
                    Real(1.0, 2.0, name="fake_weight"),  # Class weight for fake images
                    
                    # Augmentation parameters
                    Integer(10, 30, name="rotation_range"),
                    Real(0.1, 0.3, name="shift_range"),
                    Real(0.7, 0.9, name="brightness_min"),
                    Real(1.1, 1.3, name="brightness_max"),
                    Real(0.1, 0.3, name="zoom_range")
                ]
            },
            "SVM": {
                "default_params": {
                    "model_params": {
                        "C": 1.0,
                        "kernel": "rbf",
                        "probability": True
                    },
                    "training_params": {
                        "batch_size": None,
                        "epochs": None
                    }
                },
                "search_space": [
                    Real(0.1, 10.0, name="C", prior="log-uniform"),
                    Categorical(["linear", "rbf"], name="kernel")
                ]
            },
            "Random Forest": {
                "default_params": {
                    "model_params": {
                        "n_estimators": 100,
                        "max_depth": 10,
                        "min_samples_split": 2,
                        "n_jobs": -1
                    },
                    "training_params": {
                        "batch_size": None,
                        "epochs": None
                    }
                },
                "search_space": [
                    Integer(50, 200, name="n_estimators"),
                    Integer(5, 50, name="max_depth"),
                    Integer(2, 10, name="min_samples_split")
                ]
            }
        }
        
        # Active models configuration
        self.ACTIVE_MODELS = {
            "CNN": True,
            "Mesonet": False,
            "SVM": False,
            "Random Forest": False
        }

        # Create save directory if it doesn't exist
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        print(f"Save directory initialized at: {self.SAVE_DIR}")
        
    def get_model_config(self, model_name):
        """Get configuration for a specific model"""
        return self.MODEL_CONFIGS.get(model_name, {})
        
    def get_default_params(self, model_name):
        """Get default parameters for a specific model"""
        model_config = self.get_model_config(model_name)
        return model_config.get("default_params", {})
        
    def get_search_space(self, model_name):
        """Get search space for a specific model"""
        model_config = self.get_model_config(model_name)
        return model_config.get("search_space", [])
