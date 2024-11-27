# src/main.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_METAL_DEVICE_FORCE'] = '1'

from config.config import Config
from data.data_loader import DataLoader
from models.factory import ModelFactory
from training.trainer import ModelTrainer
from utils.gpu import setup_metal_gpu

def main():
    # Initialize configuration
    config = Config()
    
    # Setup Metal GPU
    gpu_available = setup_metal_gpu()
    
    if gpu_available:
        print("\nMetal GPU enabled")
    else:
        print("\nRunning on CPU only")
    
    print("\nConfiguration:")
    print(f"Dataset Fraction Enabled: {config.USE_DATASET_FRACTION}")
    if config.USE_DATASET_FRACTION:
        print(f"Dataset Fraction: {config.DATASET_FRACTION * 100}%")
    print(f"Image Size: {config.IMG_SIZE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Active Models: {[model for model, active in config.ACTIVE_MODELS.items() if active]}\n")
    
    # Initialize components
    data_loader = DataLoader(config)
    model_factory = ModelFactory(config)
    model_trainer = ModelTrainer(config, model_factory)

    # Load and prepare data
    train_data, val_data, test_data = data_loader.load_datasets()
    
    # Train and evaluate models
    model_trainer.train_and_evaluate(train_data, val_data, test_data)

if __name__ == "__main__":
    main()
