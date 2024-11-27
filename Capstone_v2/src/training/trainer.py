import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from joblib import cpu_count
import time
import traceback
from tqdm import tqdm
from models.progress_models import *
from utils.time_estimation import TimeEstimator
from utils.metrics_handler import MetricsHandler
from utils.memory_monitor import MemoryMonitor
from utils.dashboard import DashboardManager
from utils.visualization import AdvancedVisualizer

# Configure TensorFlow for Metal GPU
try:
    # Configure memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            # Set a memory limit (e.g., 4GB)
            tf.config.set_logical_device_configuration(
                device,
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
            )
        print("GPU memory growth enabled")
    else:
        print("No GPU devices found")
except Exception as e:
    print(f"Error configuring GPU: {str(e)}")
    print("Falling back to CPU")

class ModelTrainer:
    """Class to handle model training and evaluation"""
    def __init__(self, config, model_factory):
        self.config = config
        self.model_factory = model_factory
        self.trained_models = {}
        self.scalers = {}

        self.time_estimator = TimeEstimator(config_dir=os.path.join(config.SAVE_DIR, "calibration"))
        self.metrics_handler = MetricsHandler(config)

        # Initialize dashboard manager if dashboards enabled
        if self.config.SHOW_DASHBOARDS:
            self.dashboard_manager = DashboardManager(config)
            # Initialize dashboards but don't start server yet
            self.dashboard_manager.initialize_dashboards()
        else:
            self.dashboard_manager = None

        # Create necessary directories
        os.makedirs(self.config.SAVE_DIR, exist_ok=True)
        print(f"Save directory initialized at: {self.config.SAVE_DIR}")

    def train_and_evaluate(self, train_data, val_data, test_data):
        """Train and evaluate all active models"""
        try:
            # Train and evaluate each active model
            for model_name, is_active in self.config.ACTIVE_MODELS.items():
                if not is_active:
                    continue

                print(f"\nTraining {model_name}...")
                model = self._train_model(model_name, train_data, val_data)
                
                if model is not None:
                    self.trained_models[model_name] = model
                    self._evaluate_model(model, model_name, val_data, test_data)
            
            # Start dashboard server after all visualizations are generated
            if self.dashboard_manager:
                print("\nStarting dashboard server...")
                self.dashboard_manager.start_server()
                print("\nTraining complete. Press Ctrl+C to end and close dashboards.")
                print(f"Dashboards available at: http://localhost:{self.config.DASHBOARD_PORT}")
                try:
                    self.dashboard_manager.wait_for_user()
                except KeyboardInterrupt:
                    print("\nShutting down...")
                    self.dashboard_manager.stop()
            else:
                print("\nTraining complete.")
                    
        except Exception as e:
            print(f"Error in training: {str(e)}")
            if self.dashboard_manager:
                self.dashboard_manager.stop()
            raise e

    def _train_model(self, model_name, train_data, val_data):
        """Train a specific model using the configured optimization strategy"""
        try:
            from training.optimization import ModelOptimizer
            from utils.validation import ParameterValidator
            
            # Initialize optimizer
            optimizer = ModelOptimizer(self.config, self.model_factory, train_data, val_data)
            
            # Get optimized or default parameters
            params = optimizer.optimize(model_name, self.config.OPTIMIZATION_STRATEGY)
            
            # Store optimization results for visualization
            self.optimization_results = optimizer.optimization_results
            
            # Validate parameters using the ParameterValidator
            try:
                ParameterValidator.validate_parameters(model_name, params)
            except ValueError as e:
                print(f"Warning: Parameter validation failed: {str(e)}")
                print("Using default parameters...")
                params = ModelParameters.get_default_params()[model_name]
            
            # Log parameters
            ParameterValidator.log_parameters(
                model_name=model_name,
                params=params,
                optimization_strategy=self.config.OPTIMIZATION_STRATEGY,
                log_dir=self.config.SAVE_DIR
            )
            
            # Train model based on type
            if model_name in ["CNN", "Mesonet"]:
                return self._train_neural_network(model_name, params, train_data, val_data)
            else:
                return self._train_classical_model(model_name, params["model_params"], train_data)
                
        except Exception as e:
            print(f"Error in _train_model: {str(e)}")
            print(f"Model: {model_name}")
            print(f"Parameters: {params if 'params' in locals() else 'Not available'}")
            raise

    def _evaluate_model(self, model, model_name, val_data, test_data):
        """Evaluate model on validation and test sets"""
        confusion_matrices = {}  # Store confusion matrices for visualization
        
        for dataset_name, (images, labels) in [("Validation", val_data), ("Test", test_data)]:
            print(f"\nEvaluating {model_name} on {dataset_name} dataset...")
            
            try:
                # Get predictions
                if model_name in ["CNN", "Mesonet"]:
                    eval_dataset = (tf.data.Dataset.from_tensor_slices((images, labels))
                        .batch(self.config.BATCH_SIZE)
                        .prefetch(tf.data.AUTOTUNE))
                    predictions = model.predict(eval_dataset, verbose=1)
                    predictions = (predictions > 0.5).astype(int).flatten()
                else:
                    images_reshaped = images.reshape(images.shape[0], -1)
                    if model_name in self.scalers:
                        scaler = self.scalers[model_name]
                        images_standardized = scaler.transform(images_reshaped)
                    else:
                        print(f"Warning: No stored scaler found for {model_name}. Using new standardization.")
                        scaler = StandardScaler()
                        images_standardized = scaler.fit_transform(images_reshaped)
                    
                    if hasattr(self, 'pca') and model_name in ["SVM", "Random Forest"]:
                        print("Applying PCA transformation...")
                        images_standardized = self.pca.transform(images_standardized)
                    
                    predictions = model.predict(images_standardized)

                # Calculate metrics using MetricsHandler
                metrics = self.metrics_handler.calculate_metrics(labels, predictions)
                
                # Print metrics
                self.metrics_handler.print_metrics(model_name, dataset_name, metrics)
                
                # Log metrics
                self.metrics_handler.log_metrics(model_name, dataset_name, metrics)
                
                # Update dashboard if enabled
                if self.dashboard_manager:
                    self.dashboard_manager.update_metrics(model_name, metrics)
                
                # Generate and store confusion matrix
                cm = self.metrics_handler.plot_confusion_matrix(labels, predictions, model_name)
                confusion_matrices[dataset_name] = cm
                    
            except Exception as e:
                print(f"Error evaluating {model_name} on {dataset_name}: {str(e)}")
                print("Stack trace:")
                traceback.print_exc()

        # Create visualizations with confusion matrix
        try:
            visualizer = AdvancedVisualizer(self.config.SAVE_DIR)
            
            # Create standalone confusion matrix plot
            if "Test" in confusion_matrices:
                visualizer.create_confusion_matrix_plot(
                    confusion_matrices["Test"],
                    model_name
                )
            
            # Create summary dashboard with confusion matrix
            if hasattr(self, 'optimization_results'):
                visualizer.create_summary_dashboard(
                    model_name,
                    self.optimization_results,
                    confusion_matrix=confusion_matrices.get("Test")
                )
                
            # Update dashboards after generating new visualizations
            if self.dashboard_manager:
                self.dashboard_manager.update_dashboards()
                
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            traceback.print_exc()

    def _train_neural_network(self, model_name, params, train_data, val_data):
        """Train neural network models with proper parameters"""
        try:
            # Data preparation
            train_images = tf.convert_to_tensor(train_data[0], dtype=tf.float32)
            train_labels = tf.convert_to_tensor(train_data[1], dtype=tf.float32)
            val_images = tf.convert_to_tensor(val_data[0], dtype=tf.float32)
            val_labels = tf.convert_to_tensor(val_data[1], dtype=tf.float32)
            
            train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

            if params["training_params"].get("use_augmentation", False):
                def augment_data(image, label):
                    image = tf.cast(image, tf.float32)
                    image = tf.image.random_flip_left_right(image)
                    image = tf.image.random_brightness(image, 0.1)
                    image = tf.image.random_contrast(image, 0.9, 1.1)
                    return image, label

                train_dataset = (train_dataset
                    .shuffle(10000)
                    .map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(params["training_params"]["batch_size"])
                    .prefetch(tf.data.AUTOTUNE))
            else:
                train_dataset = (train_dataset
                    .shuffle(10000)
                    .batch(params["training_params"]["batch_size"])
                    .prefetch(tf.data.AUTOTUNE))

            val_dataset = (val_dataset
                .batch(params["training_params"]["batch_size"])
                .prefetch(tf.data.AUTOTUNE))

            # Create model with model_params
            model = self.model_factory.create_model(model_name, {"model_params": params["model_params"]})
            
            if model is None:
                raise ValueError(f"Failed to create {model_name} model")

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(self.config.SAVE_DIR, f'best_{model_name.lower()}.keras'),
                    save_best_only=True,
                    monitor='val_accuracy',
                    mode='max'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]

            print("\nStarting training...")
            print(f"Training dataset size: {train_data[0].shape}")
            print(f"Validation dataset size: {val_data[0].shape}")
            print(f"Batch size: {params['training_params']['batch_size']}")
            print(f"Epochs: {params['training_params']['epochs']}")

            history = model.fit(
                train_dataset,
                epochs=params["training_params"]["epochs"],
                validation_data=val_dataset,
                callbacks=callbacks,
                class_weight=params["training_params"].get("class_weights", None),
                verbose=1
            )
            
            return model
            
        except Exception as e:
            print(f"Error in _train_neural_network: {str(e)}")
            print(f"Model name: {model_name}")
            print(f"Params: {params}")
            traceback.print_exc()
            raise e

    def _train_classical_model(self, model_name, params, train_data):
        """Train classical ML models with optimizations and progress monitoring"""
        try:
            print(f"\nTraining {model_name} model...")
            
            # Data preparation with progress bar
            print("\nPreparing data...")
            with tqdm(total=3, desc="Data preparation") as pbar:
                train_images = train_data[0].reshape(train_data[0].shape[0], -1)
                pbar.update(1)
                
                scaler = StandardScaler()
                train_images = scaler.fit_transform(train_images)
                pbar.update(1)
                
                train_labels = train_data[1]
                pbar.update(1)
            
            self.scalers[model_name] = scaler
            n_samples, n_features = train_images.shape
            print(f"\nInput shape: {train_images.shape}")
            print(f"Parameters: {params}")
            
            # Estimate memory requirements
            estimated_memory = MemoryMonitor.estimate_memory_requirement(n_samples, n_features, model_name)
            print(f"Estimated memory requirement: {estimated_memory}")
            
            if model_name == "SVM":
                return self._train_svm(train_images, train_labels, params, n_samples, n_features)
            elif model_name == "Random Forest":
                return self._train_random_forest(train_images, train_labels, params, n_samples, n_features)
            else:
                raise ValueError(f"Unknown model type: {model_name}")
                
        except Exception as e:
            print(f"Error in _train_classical_model: {str(e)}")
            print(f"Model name: {model_name}")
            print(f"Params: {params}")
            traceback.print_exc()
            raise e

    def _train_svm(self, train_images, train_labels, params, n_samples, n_features):
        """Train SVM with optimization options and progress monitoring"""
        try:
            # Get estimate from TimeEstimator
            estimate = self.time_estimator.estimate_svm_time(n_samples, n_features)
            print(f"\nEstimated training time: {estimate['formatted']}")
            print(f"Expected range: {estimate['range']['low']} - {estimate['range']['high']}")
            
            while True:
                try:
                    choice = self._get_optimization_choice("SVM")
                    
                    start_mem = MemoryMonitor.get_current_usage()
                    start_time = time.time()
                    
                    if choice == "2":  # PCA
                        train_images = self._apply_pca(train_images, n_samples)
                    elif choice == "3":  # LinearSVC
                        return self._train_linear_svc(train_images, train_labels, params, n_samples, n_features)
                    elif choice == "4":  # Subset
                        try:
                            train_images, train_labels = self._get_data_subset(train_images, train_labels)
                        except KeyboardInterrupt:
                            print("\nTraining cancelled by user")
                            return None
                    elif choice not in ["1", "2", "3", "4"]:
                        print("Invalid choice. Please try again.")
                        continue
                    
                    # Proceed with training
                    print("\nStarting SVM training...")
                    model = ProgressSVC(**params)
                    model.fit(train_images, train_labels)
                    
                    # Record actual time
                    training_time = time.time() - start_time
                    self.time_estimator.record_actual_time("svm", training_time)
                    
                    self._print_training_summary("SVM", start_time, start_mem)
                    return model
                    
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user")
                    user_input = input("Do you want to try a different optimization? (y/n): ").strip().lower()
                    if user_input not in ['y', 'yes']:
                        raise KeyboardInterrupt("Training cancelled by user")
                        
        except Exception as e:
            print(f"Error in SVM training: {str(e)}")
            raise

    def _train_random_forest(self, train_images, train_labels, params, n_samples, n_features):
        """Train Random Forest with optimization options and progress monitoring"""
        n_jobs = self._get_n_jobs(params)
        n_trees = params.get('n_estimators', 100)
        
        # Get estimate from TimeEstimator
        estimate = self.time_estimator.estimate_rf_time(
            n_samples, n_features, n_trees, n_jobs
        )
        print(f"\nEstimated training time: {estimate['formatted']}")
        print(f"Expected range: {estimate['range']['low']} - {estimate['range']['high']}")
        
        print(f"\nRandom Forest Configuration:")
        print(f"Number of trees: {n_trees}")
        print(f"CPU cores to be used: {n_jobs}")
        
        choice = self._get_optimization_choice("Random Forest")
        
        start_mem = MemoryMonitor.get_current_usage()
        start_time = time.time()
        
        if choice == "2":  # PCA
            train_images = self._apply_pca(train_images, n_samples)
        elif choice == "3":  # Subset
            train_images, train_labels = self._get_data_subset(train_images, train_labels)
        elif choice == "4":  # Adjust trees
            params = self._adjust_n_trees(params)
        
        model = ProgressRandomForest(**params)
        model.fit(train_images, train_labels)
        
        # Record actual time
        training_time = time.time() - start_time
        self.time_estimator.record_actual_time("random_forest", training_time)
        
        self._print_training_summary("Random Forest", start_time, start_mem, 
                                n_trees=params['n_estimators'])
        
        self._analyze_feature_importance(model, train_images)
        return model

    def _get_n_jobs(self, params):
        """Get number of jobs for parallel processing with better system awareness"""
        try:
            n_jobs = params.get('n_jobs', -1)
            available_cpus = cpu_count()
            
            if n_jobs < 0:
                # Convert negative n_jobs to actual core count
                n_jobs = max(1, available_cpus + 1 + n_jobs)
                
            # Don't exceed available cores and leave one core free for system
            n_jobs = min(n_jobs, max(1, available_cpus - 1))
            
            print(f"CPU cores detected: {available_cpus}")
            print(f"Using {n_jobs} cores for parallel processing")
            
            return n_jobs
            
        except Exception as e:
            print(f"Warning: Error detecting CPU count ({str(e)}). Using 1 core.")
            return 1

    def _get_optimization_choice(self, model_type):
        print("\nOptimization options:")
        print("1. Continue with full dataset")
        print("2. Use PCA dimension reduction")
        if model_type == "SVM":
            print("3. Use Linear SVC (faster but only for linear kernel)")
        print(f"{'4' if model_type == 'SVM' else '3'}. Use subset of data")
        if model_type == "Random Forest":
            print("4. Adjust number of trees")
        
        return input(f"Choose optimization (1-{4 if model_type == 'Random Forest' else 3}): ")

    def _apply_pca(self, train_images, n_samples):
        n_components = min(n_samples, 1000)
        print(f"\nReducing dimensions from {train_images.shape[1]} to {n_components} using PCA...")
        with tqdm(total=1, desc="PCA reduction") as pbar:
            pca = PCA(n_components=n_components)
            train_images = pca.fit_transform(train_images)
            pbar.update(1)
        print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
        self.pca = pca
        return train_images

    def _get_data_subset(self, train_images, train_labels):
        """Get a subset of the training data with robust input handling"""
        while True:
            try:
                # Show current data size
                print(f"\nCurrent dataset size: {len(train_images)} samples")
                
                # Suggest reasonable default sizes
                suggested_size = min(2000, len(train_images) // 2)
                user_input = input(f"Enter subset size (1-{len(train_images)}, default={suggested_size}, press 'q' to quit): ").strip()
                
                # Handle quit option
                if user_input.lower() == 'q':
                    print("Exiting training...")
                    raise KeyboardInterrupt("User cancelled training")
                
                # Use default if empty input
                if user_input == '':
                    subset_size = suggested_size
                    print(f"Using default size: {subset_size}")
                else:
                    subset_size = int(user_input)
                
                # Validate input range
                if subset_size <= 0 or subset_size > len(train_images):
                    print(f"Error: Size must be between 1 and {len(train_images)}")
                    continue
                
                # Ask for confirmation
                confirm = input(f"Confirm using {subset_size} samples? (y/n, default=y): ").strip().lower()
                if confirm in ['', 'y', 'yes']:
                    indices = np.random.choice(len(train_images), subset_size, replace=False)
                    return train_images[indices], train_labels[indices]
                
                # If not confirmed, loop back
                print("Okay, let's try again...")
                
            except ValueError as e:
                print("Error: Please enter a valid number")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                print("Let's try again...")

    def _print_training_summary(self, model_type, start_time, start_mem, **kwargs):
        end_time = time.time()
        end_mem = MemoryMonitor.get_current_usage()
        training_time = end_time - start_time
        
        print(f"\n{model_type} Training Summary:")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Memory usage: {start_mem} → {end_mem}")
        
        if "n_trees" in kwargs:
            trees_per_second = kwargs["n_trees"] / training_time
            print(f"Trees built: {kwargs['n_trees']}")
            print(f"Trees per second: {trees_per_second:.2f}")

    def _analyze_feature_importance(self, model, train_images):
        print("\nAnalyzing feature importance...")
        with tqdm(total=2, desc="Feature analysis") as pbar:
            importances = pd.Series(
                model.feature_importances_,
                index=[f"feature_{i}" for i in range(train_images.shape[1])]
            ).sort_values(ascending=False)
            pbar.update(1)
            
            plt.figure(figsize=(10, 5))
            importances.head(20).plot(kind='bar')
            plt.title('Top 20 Most Important Features')
            plt.tight_layout()
            plt.show()
            pbar.update(1)
        
        print("\nTop 10 Most Important Features:")
        print(importances.head(10))
