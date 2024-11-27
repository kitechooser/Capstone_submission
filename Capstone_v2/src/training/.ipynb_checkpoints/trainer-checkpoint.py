# src/training/trainer.py

import os
import csv
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    """Class to handle model training and evaluation"""
    def __init__(self, config, model_factory):
        self.config = config
        self.model_factory = model_factory
        self.trained_models = {}
    
        os.makedirs(self.config.SAVE_DIR, exist_ok=True)
    
        if not os.path.exists(self.config.LOG_CSV):
            with open(self.config.LOG_CSV, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Model', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1'])
    
        print("ModelTrainer initialized with Metal configuration:")
        print(f"- Training augmentation enabled")
        print(f"- Save directory: {self.config.SAVE_DIR}")
        print(f"- Metrics log file: {self.config.LOG_CSV}")
        print(f"- Image size: {self.config.IMG_SIZE}")
        print(f"- Batch size: {self.config.BATCH_SIZE}")

    def train_and_evaluate(self, train_data, val_data, test_data):
        """Train and evaluate all active models"""
        for model_name, is_active in self.config.ACTIVE_MODELS.items():
            if not is_active:
                continue

            print(f"\nTraining {model_name}...")
            model = self._train_model(model_name, train_data, val_data)
            self.trained_models[model_name] = model
            
            for dataset_name, (images, labels) in [("Validation", val_data), ("Test", test_data)]:
                self._evaluate_and_log(model, model_name, images, labels, dataset_name)

        self._print_confusion_matrices(test_data)

    def _train_model(self, model_name, train_data, val_data):
        """Train a specific model using the configured optimization strategy"""
        from models.parameters import ModelParameters
        params = ModelParameters.get_default_params()[model_name]
        
        if self.config.OPTIMIZATION_STRATEGY == "None":
            if model_name in ["CNN", "Mesonet"]:
                return self._train_neural_network(model_name, params, train_data, val_data)
            else:
                return self._train_classical_model(model_name, params["model_params"], train_data)
        
        return self._train_neural_network(model_name, params, train_data, val_data)

    def _train_neural_network(self, model_name, params, train_data, val_data):
        """Train neural network models optimized for Metal"""
        try:
            with tf.device('/CPU:0'):
                train_images = tf.convert_to_tensor(train_data[0], dtype=tf.float32)
                train_labels = tf.convert_to_tensor(train_data[1], dtype=tf.float32)
                val_images = tf.convert_to_tensor(val_data[0], dtype=tf.float32)
                val_labels = tf.convert_to_tensor(val_data[1], dtype=tf.float32)
                
                train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
                val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    
                if params["training_params"].get("use_augmentation", False):
                    print("Using data augmentation with prefetch optimization")
                    
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
    
            model = self.model_factory.create_model(model_name, params["model_params"])
            
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
            import traceback
            traceback.print_exc()
            raise e

    def _train_classical_model(self, model_name, params, train_data):
        """Train classical ML models (SVM, Random Forest)"""
        try:
            print(f"\nTraining {model_name} model...")
            
            # Reshape the image data to 2D (samples, features)
            train_images = train_data[0].reshape(train_data[0].shape[0], -1)
            train_labels = train_data[1]
            
            # Create and train the model
            model = self.model_factory.create_model(model_name, params)
            
            print(f"Input shape: {train_images.shape}")
            print(f"Parameters: {params}")
            
            # Standardize the features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            train_images = scaler.fit_transform(train_images)
            
            # Train the model
            model.fit(train_images, train_labels)
            
            print(f"{model_name} training completed")
            return model
            
        except Exception as e:
            print(f"Error in _train_classical_model: {str(e)}")
            print(f"Model name: {model_name}")
            print(f"Params: {params}")
            import traceback
            traceback.print_exc()
            raise e
    
    # Also update the _evaluate_and_log method to handle classical models better
    def _evaluate_and_log(self, model, model_name, images, labels, dataset_name):
        """Evaluate model with prefetch optimization"""
        print(f"\nEvaluating {model_name} on {dataset_name} dataset...")
        
        if model_name in ["CNN", "Mesonet"]:
            eval_dataset = (tf.data.Dataset.from_tensor_slices((images, labels))
                .batch(self.config.BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))
            predictions = model.predict(eval_dataset, verbose=1)
            predictions = (predictions > 0.5).astype(int).flatten()
        else:
            # For classical models, reshape and standardize the data
            images_reshaped = images.reshape(images.shape[0], -1)
            scaler = StandardScaler()
            images_standardized = scaler.fit_transform(images_reshaped)
            predictions = model.predict(images_standardized)
    
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0)
        }
    
        print(f"\n{model_name} Performance on {dataset_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name.capitalize()}: {value:.4f}")
    
        with open(self.config.LOG_CSV, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_name, dataset_name] + list(metrics.values()))
    
    # Also update _print_confusion_matrices to handle classical models
    def _print_confusion_matrices(self, test_data):
        """Print and visualize confusion matrices for all trained models"""
        for model_name, model in self.trained_models.items():
            print(f"\nGenerating confusion matrix for {model_name}...")
            
            if model_name in ["CNN", "Mesonet"]:
                predictions = (model.predict(test_data[0]) > 0.5).astype(int).flatten()
            else:
                # For classical models, reshape and standardize the data
                images_reshaped = test_data[0].reshape(test_data[0].shape[0], -1)
                scaler = StandardScaler()
                images_standardized = scaler.fit_transform(images_reshaped)
                predictions = model.predict(images_standardized)
            
            cm = confusion_matrix(test_data[1], predictions)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Fake', 'Real'],
                       yticklabels=['Fake', 'Real'])
            plt.title(f"Confusion Matrix - {model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.savefig(os.path.join(self.config.SAVE_DIR, f'{model_name.lower()}_confusion_matrix.png'))
            plt.show()
