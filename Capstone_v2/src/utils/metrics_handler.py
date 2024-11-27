import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import json
from datetime import datetime

class MetricsHandler:
    """Handles metrics calculation, logging, and visualization"""
    
    def __init__(self, config):
        self.config = config
        self._initialize_metrics_log()
        self._initialize_metrics_json()
        self.confusion_matrices = {}
        
    def _initialize_metrics_log(self):
        """Initialize metrics log file if it doesn't exist"""
        headers = ['Timestamp', 'Model', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1']
        
        # Create metrics.csv in the main save directory
        metrics_csv = os.path.join(self.config.SAVE_DIR, 'model_metrics.csv')
        if not os.path.exists(metrics_csv):
            with open(metrics_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                print(f"Created metrics log file with headers: {headers}")

    def _initialize_metrics_json(self):
        """Initialize metrics JSON file"""
        # Create metrics.json in the main save directory
        metrics_file = os.path.join(self.config.SAVE_DIR, 'metrics.json')
        
        if not os.path.exists(metrics_file):
            metrics_data = {}
            for model_name, is_active in self.config.ACTIVE_MODELS.items():
                if is_active:
                    metrics_data[model_name] = {
                        'accuracy': [],
                        'precision': [],
                        'recall': [],
                        'f1': [],
                        'timestamps': []
                    }
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            print(f"Created metrics JSON file: {metrics_file}")
    
    def calculate_metrics(self, true_labels, predictions):
        """Calculate all metrics for a model's predictions"""
        try:
            metrics = {
                'accuracy': accuracy_score(true_labels, predictions),
                'precision': precision_score(true_labels, predictions, zero_division=1),
                'recall': recall_score(true_labels, predictions, zero_division=1),
                'f1': f1_score(true_labels, predictions, zero_division=1)
            }
            
            # Validate metrics
            for metric_name, value in metrics.items():
                if not (isinstance(value, (int, float)) and 0 <= value <= 1):
                    print(f"Warning: Invalid {metric_name} value: {value}")
                    metrics[metric_name] = float(value) if isinstance(value, (int, float)) else 0.0
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
    
    def log_metrics(self, model_name, dataset_name, metrics):
        """Log metrics to CSV file and JSON"""
        timestamp = datetime.now().isoformat()
        
        # Log to CSV in main save directory
        metrics_csv = os.path.join(self.config.SAVE_DIR, 'model_metrics.csv')
        with open(metrics_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, model_name, dataset_name] + list(metrics.values()))
        
        # Log to JSON in main save directory
        metrics_file = os.path.join(self.config.SAVE_DIR, 'metrics.json')
        try:
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
            else:
                metrics_data = {}
            
            if model_name not in metrics_data:
                metrics_data[model_name] = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'timestamps': []
                }
            
            # Update metrics and timestamp
            for metric_name, value in metrics.items():
                # Convert value to float and validate
                try:
                    float_value = float(value)
                    if 0 <= float_value <= 1:
                        metrics_data[model_name][metric_name].append(float_value)
                    else:
                        print(f"Warning: Invalid {metric_name} value: {float_value}")
                        metrics_data[model_name][metric_name].append(0.0)
                except (TypeError, ValueError) as e:
                    print(f"Error converting {metric_name} value: {value}")
                    metrics_data[model_name][metric_name].append(0.0)
                    
            metrics_data[model_name]['timestamps'].append(timestamp)
            
            # Save updated metrics
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            print(f"\nUpdated metrics for {model_name} ({dataset_name}):")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
                
        except Exception as e:
            print(f"Error updating metrics JSON: {str(e)}")
            raise
    
    def print_metrics(self, model_name, dataset_name, metrics):
        """Print metrics in a formatted way"""
        print(f"\n{model_name} Performance on {dataset_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name.capitalize()}: {value:.4f}")
    
    def plot_confusion_matrix(self, true_labels, predictions, model_name):
        """Create and save confusion matrix plot"""
        cm = confusion_matrix(true_labels, predictions)
        self.confusion_matrices[model_name] = cm
        
        # Save confusion matrix data in main save directory
        cm_file = os.path.join(self.config.SAVE_DIR, f'{model_name.lower()}_confusion_matrix.json')
        with open(cm_file, 'w') as f:
            json.dump({
                'matrix': cm.tolist(),
                'labels': ['Fake', 'Real'],
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Create visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Fake', 'Real'],
                   yticklabels=['Fake', 'Real'])
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        # Add timestamp
        plt.figtext(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                   ha='right', va='bottom', fontsize=8, color='gray')
        
        # Save plot in main save directory
        save_path = os.path.join(self.config.SAVE_DIR, f'{model_name.lower()}_confusion_matrix.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Print confusion matrix
        print(f"\nConfusion Matrix for {model_name}:")
        print("             Predicted")
        print("             Fake  Real")
        print(f"Actual Fake  {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"      Real  {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        return cm
