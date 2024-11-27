# src/utils/validation.py

from typing import Dict, Any
import logging
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ParameterValidator:
    """Validates and logs model parameters"""
    
    @staticmethod
    def get_required_parameters():
        """Define required parameters for each model type"""
        return {
            "CNN": {
                "model_params": [
                    "learning_rate", "num_conv_layers", "num_filters",
                    "kernel_size", "pool_type", "dropout_rate"
                ],
                "training_params": [
                    "batch_size", "epochs", "use_augmentation"
                ]
            },
            "Mesonet": {
                "model_params": [
                    "learning_rate", "dropout_rate"
                ],
                "training_params": [
                    "batch_size", "epochs", "use_augmentation", "class_weights"
                ]
            },
            "SVM": {
                "model_params": ["C", "kernel"],
                "training_params": []
            },
            "Random Forest": {
                "model_params": [
                    "n_estimators", "max_depth", "min_samples_split"
                ],
                "training_params": []
            }
        }

    @staticmethod
    def validate_parameters(model_name: str, params: Dict[str, Dict[str, Any]]) -> bool:
        """
        Validate that all required parameters are present
        Returns True if valid, raises ValueError if not
        """
        required_params = ParameterValidator.get_required_parameters()
        
        if model_name not in required_params:
            raise ValueError(f"Unknown model type: {model_name}")
            
        for param_type in ["model_params", "training_params"]:
            if param_type not in params:
                raise ValueError(f"Missing {param_type} in parameters")
                
            for required_param in required_params[model_name][param_type]:
                if required_param not in params[param_type]:
                    raise ValueError(
                        f"Missing required parameter '{required_param}' in {param_type} "
                        f"for model {model_name}"
                    )
        
        return True

    @staticmethod
    def _convert_to_serializable(obj):
        """Convert numpy/numeric types to standard Python types"""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                          np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: ParameterValidator._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ParameterValidator._convert_to_serializable(x) for x in obj]
        return obj

    @staticmethod
    def log_parameters(model_name: str, params: Dict[str, Dict[str, Any]], 
                      optimization_strategy: str, log_dir: str) -> None:
        """Log parameters to file and console"""
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{model_name.lower()}_parameters.json")
        
        # Convert parameters to serializable format
        serializable_params = ParameterValidator._convert_to_serializable(params)
        
        # Add metadata to parameters
        params_with_meta = {
            "model_name": model_name,
            "optimization_strategy": optimization_strategy,
            "timestamp": pd.Timestamp.now().isoformat(),
            "parameters": serializable_params
        }
        
        # Log to file
        with open(log_file, 'w') as f:
            json.dump(params_with_meta, f, indent=4)
            
        # Log to console
        print(f"\nParameters for {model_name} (Optimization: {optimization_strategy}):")
        print("\nModel Parameters:")
        for key, value in serializable_params["model_params"].items():
            print(f"  {key}: {value}")
        print("\nTraining Parameters:")
        for key, value in serializable_params["training_params"].items():
            print(f"  {key}: {value}")
