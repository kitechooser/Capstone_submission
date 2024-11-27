import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import optuna
from utils.validation import ParameterValidator
from utils.visualization import AdvancedVisualizer

class ModelOptimizer:
    """Class to handle model parameter optimization"""
    
    def __init__(self, config, model_factory, train_data, val_data):
        self.config = config
        self.model_factory = model_factory
        self.train_data = train_data
        self.val_data = val_data
        self.optimization_results = {"trials": []}
        self.visualizer = AdvancedVisualizer(config.SAVE_DIR)

    def optimize(self, model_name, strategy="None"):
        """Optimize model parameters using specified strategy"""
        print(f"\nStarting optimization for {model_name} using {strategy} strategy")
        
        if strategy == "None":
            return self.config.get_default_params(model_name)
        elif strategy == "BO":
            return self._bayesian_optimization(model_name)
        elif strategy == "Optuna":
            return self._optuna_optimization(model_name)
        else:
            print(f"Unknown optimization strategy: {strategy}")
            return self.config.get_default_params(model_name)

    def _bayesian_optimization(self, model_name):
        """Perform Bayesian optimization"""
        search_space = self.config.get_search_space(model_name)
        if not search_space:
            print(f"No search space defined for {model_name}")
            return self.config.get_default_params(model_name)
            
        print(f"Number of trials: {self.config.OPTIMIZATION_TRIALS}")
        print(f"Number of random starts: {self.config.OPTIMIZATION_RANDOM_STARTS}")
        
        def objective(x):
            # Convert parameters to dictionary
            param_dict = {dim.name: val for dim, val in zip(search_space, x)}
            
            # Print current parameters
            print("\nTrial Parameters:")
            for name, value in param_dict.items():
                print(f"  {name}: {value}")
            
            if model_name == "Mesonet":
                # Handle special parameters for Mesonet
                model_params = {
                    k: v for k, v in param_dict.items() 
                    if k in ['initial_filters', 'conv1_size', 'conv_other_size', 
                            'dropout_rate', 'dense_units', 'leaky_relu_alpha', 
                            'learning_rate', 'batch_size']
                }
                
                # Create full parameters dictionary
                params = {
                    "model_params": model_params,
                    "training_params": {
                        "batch_size": param_dict.get('batch_size', self.config.BATCH_SIZE),
                        "epochs": self.config.EPOCH_COUNT,
                        "use_augmentation": self.config.USE_AUGMENTATION,
                        "class_weights": {0: param_dict.get('fake_weight', 1.5), 1: 1.0},
                        "augmentation_params": {
                            "rotation_range": param_dict.get('rotation_range', 20),
                            "width_shift_range": param_dict.get('shift_range', 0.2),
                            "height_shift_range": param_dict.get('shift_range', 0.2),
                            "brightness_range": [
                                param_dict.get('brightness_min', 0.8),
                                param_dict.get('brightness_max', 1.2)
                            ],
                            "horizontal_flip": True,
                            "zoom_range": param_dict.get('zoom_range', 0.2)
                        }
                    }
                }
            elif model_name == "CNN":
                # Handle kernel_size conversion for CNN
                if 'kernel_size' in param_dict:
                    if param_dict['kernel_size'] == "3x3":
                        param_dict['kernel_size'] = (3, 3)
                    elif param_dict['kernel_size'] == "5x5":
                        param_dict['kernel_size'] = (5, 5)
                
                # Create full parameters dictionary
                params = {
                    "model_params": param_dict,
                    "training_params": self.config.get_default_params(model_name)["training_params"]
                }
            else:
                # For other models, use parameters as-is
                params = {
                    "model_params": param_dict,
                    "training_params": self.config.get_default_params(model_name)["training_params"]
                }
            
            try:
                # Create and train model
                model = self.model_factory.create_model(model_name, params)
                if model is None:
                    print(f"Error creating model {model_name}")
                    print(f"Params: {params}")
                    return 1.0
                
                # Train model
                history = model.fit(
                    self.train_data[0], self.train_data[1],
                    validation_data=(self.val_data[0], self.val_data[1]),
                    epochs=params["training_params"]["epochs"],
                    batch_size=params["training_params"]["batch_size"],
                    verbose=0
                )
                
                # Get validation loss
                val_loss = min(history.history['val_loss'])
                
                # Store trial results
                self.optimization_results["trials"].append({
                    "parameters": param_dict,
                    "metric": float(val_loss)
                })
                
                print(f"Validation Loss: {val_loss:.4f}")
                return float(val_loss)
                
            except Exception as e:
                print(f"Error in objective function: {str(e)}")
                return 1.0

        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=self.config.OPTIMIZATION_TRIALS,
            n_random_starts=self.config.OPTIMIZATION_RANDOM_STARTS,
            noise=1e-10,
            random_state=42
        )
        
        # Convert best parameters to dictionary
        best_params = {dim.name: val for dim, val in zip(search_space, result.x)}
        
        # Get the best parameters in the same format as used in the objective function
        if model_name == "Mesonet":
            model_params = {
                k: v for k, v in best_params.items() 
                if k in ['initial_filters', 'conv1_size', 'conv_other_size', 
                        'dropout_rate', 'dense_units', 'leaky_relu_alpha', 
                        'learning_rate', 'batch_size']
            }
            
            params = {
                "model_params": model_params,
                "training_params": {
                    "batch_size": best_params.get('batch_size', self.config.BATCH_SIZE),
                    "epochs": self.config.EPOCH_COUNT,
                    "use_augmentation": self.config.USE_AUGMENTATION,
                    "class_weights": {0: best_params.get('fake_weight', 1.5), 1: 1.0},
                    "augmentation_params": {
                        "rotation_range": best_params.get('rotation_range', 20),
                        "width_shift_range": best_params.get('shift_range', 0.2),
                        "height_shift_range": best_params.get('shift_range', 0.2),
                        "brightness_range": [
                            best_params.get('brightness_min', 0.8),
                            best_params.get('brightness_max', 1.2)
                        ],
                        "horizontal_flip": True,
                        "zoom_range": best_params.get('zoom_range', 0.2)
                    }
                }
            }
        elif model_name == "CNN":
            # Handle kernel_size conversion for CNN
            if 'kernel_size' in best_params:
                if best_params['kernel_size'] == "3x3":
                    best_params['kernel_size'] = (3, 3)
                elif best_params['kernel_size'] == "5x5":
                    best_params['kernel_size'] = (5, 5)
            
            params = {
                "model_params": best_params,
                "training_params": self.config.get_default_params(model_name)["training_params"]
            }
        else:
            params = {
                "model_params": best_params,
                "training_params": self.config.get_default_params(model_name)["training_params"]
            }
        
        # Perform analysis
        self._perform_analysis(model_name)
        
        return params

    def _optuna_optimization(self, model_name):
        """Perform Optuna optimization"""
        print(f"Number of trials: {self.config.OPTIMIZATION_TRIALS}")
        
        def objective(trial):
            # Get search space from config
            search_space = self.config.get_search_space(model_name)
            param_dict = {}
            
            # Create parameters based on search space definitions
            for dim in search_space:
                if isinstance(dim, Categorical):
                    param_dict[dim.name] = trial.suggest_categorical(dim.name, dim.categories)
                elif isinstance(dim, Integer):
                    param_dict[dim.name] = trial.suggest_int(dim.name, dim.low, dim.high)
                elif isinstance(dim, Real):
                    if getattr(dim, 'prior', None) == 'log-uniform':
                        param_dict[dim.name] = trial.suggest_float(dim.name, dim.low, dim.high, log=True)
                    else:
                        param_dict[dim.name] = trial.suggest_float(dim.name, dim.low, dim.high)
            
            # Print current parameters
            print("\nTrial Parameters:")
            for name, value in param_dict.items():
                print(f"  {name}: {value}")
            
            if model_name == "Mesonet":
                # Handle special parameters for Mesonet
                model_params = {
                    k: v for k, v in param_dict.items() 
                    if k in ['initial_filters', 'conv1_size', 'conv_other_size', 
                            'dropout_rate', 'dense_units', 'leaky_relu_alpha', 
                            'learning_rate', 'batch_size']
                }
                
                # Create full parameters dictionary
                params = {
                    "model_params": model_params,
                    "training_params": {
                        "batch_size": param_dict.get('batch_size', self.config.BATCH_SIZE),
                        "epochs": self.config.EPOCH_COUNT,
                        "use_augmentation": self.config.USE_AUGMENTATION,
                        "class_weights": {0: param_dict.get('fake_weight', 1.5), 1: 1.0},
                        "augmentation_params": {
                            "rotation_range": param_dict.get('rotation_range', 20),
                            "width_shift_range": param_dict.get('shift_range', 0.2),
                            "height_shift_range": param_dict.get('shift_range', 0.2),
                            "brightness_range": [
                                param_dict.get('brightness_min', 0.8),
                                param_dict.get('brightness_max', 1.2)
                            ],
                            "horizontal_flip": True,
                            "zoom_range": param_dict.get('zoom_range', 0.2)
                        }
                    }
                }
            else:
                # Create full parameters dictionary
                params = {
                    "model_params": param_dict,
                    "training_params": self.config.get_default_params(model_name)["training_params"]
                }
            
            try:
                # Create and train model
                model = self.model_factory.create_model(model_name, params)
                if model is None:
                    print(f"Error creating model {model_name}")
                    print(f"Params: {params}")
                    return 1.0
                
                # Train model
                history = model.fit(
                    self.train_data[0], self.train_data[1],
                    validation_data=(self.val_data[0], self.val_data[1]),
                    epochs=params["training_params"]["epochs"],
                    batch_size=params["training_params"]["batch_size"],
                    verbose=0
                )
                
                # Get validation loss
                val_loss = min(history.history['val_loss'])
                
                # Store trial results
                self.optimization_results["trials"].append({
                    "parameters": param_dict,
                    "metric": float(val_loss)
                })
                
                print(f"Validation Loss: {val_loss:.4f}")
                return float(val_loss)
                
            except Exception as e:
                print(f"Error in objective function: {str(e)}")
                return 1.0

        # Create study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.config.OPTIMIZATION_TRIALS)
        
        # Get best parameters
        best_params = study.best_params
        
        # Create full parameters dictionary
        params = {
            "model_params": best_params,
            "training_params": self.config.get_default_params(model_name)["training_params"]
        }
        
        # Perform analysis
        self._perform_analysis(model_name)
        
        return params

    def _perform_analysis(self, model_name):
        """Perform analysis on optimization results"""
        print("\nPerforming optimization analysis...")
        
        if not self.optimization_results["trials"]:
            print("Warning: No trials data available for analysis")
            return
            
        try:
            # Create correlation matrix visualization
            self.visualizer.create_correlation_matrix(
                self.optimization_results["trials"],
                model_name
            )
            
            # Create parameter space visualization
            self.visualizer.create_parameter_space_visualization(
                self.optimization_results["trials"],
                model_name
            )
            
            # Create parallel coordinates plot
            self.visualizer.create_parallel_coordinates(
                self.optimization_results["trials"],
                model_name
            )
            
            # Create optimization surface plot
            self.visualizer.create_optimization_surface(
                self.optimization_results["trials"],
                model_name
            )
            
            # Create summary dashboard
            self.visualizer.create_summary_dashboard(
                model_name,
                self.optimization_results
            )
            
            # Save optimization statistics
            stats = {
                'best_metric': min([trial['metric'] for trial in self.optimization_results["trials"]]),
                'num_trials': len(self.optimization_results["trials"]),
                'parameter_importance': self._calculate_parameter_importance()
            }
            
            import json
            import os
            stats_file = os.path.join(self.config.SAVE_DIR, f'{model_name.lower()}_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
            print(f"Saved optimization statistics to: {stats_file}")
            
            print("\nOptimization Analysis Summary:")
            print(f"Best metric achieved: {stats['best_metric']:.4f}")
            print("\nParameter Importance:")
            for param, importance in stats['parameter_importance'].items():
                print(f"  {param}: {importance:.4f}")
            print("\nVisualization files have been created in the output directory")
            
        except Exception as e:
            print(f"Warning: Analysis failed: {str(e)}")

    def _calculate_parameter_importance(self):
        """Calculate parameter importance based on correlation with metric"""
        if not self.optimization_results["trials"]:
            return {}
            
        import pandas as pd
        
        # Convert trials to DataFrame
        df = pd.DataFrame([{
            **{k: v for k, v in trial['parameters'].items()},
            'metric': trial['metric']
        } for trial in self.optimization_results["trials"]])
        
        # Calculate correlation with metric
        correlations = df.corr()['metric'].abs()
        correlations = correlations.drop('metric')
        
        return correlations.to_dict()
