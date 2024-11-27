import json
import os
import time
import numpy as np
from pathlib import Path
import psutil
from datetime import datetime
from multiprocessing import cpu_count as mp_cpu_count
from joblib import cpu_count as joblib_cpu_count

class TimeEstimator:
    """Handles time estimation and calibration for model training"""
    
    def __init__(self, config_dir="./calibration"):
        self.config_dir = Path(config_dir)
        # Create all parent directories if they don't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_file = self.config_dir / "time_calibration.json"
        
        # Create initial calibration file if it doesn't exist
        if not self.calibration_file.exists():
            with open(self.calibration_file, 'w') as f:
                json.dump({"svm": [], "random_forest": []}, f)
                
        self.history = self._load_calibration()
        self.current_run = {}

    def _load_calibration(self):
        """Load historical calibration data"""
        if self.calibration_file.exists():
            try:
                with open(self.calibration_file, 'r') as f:
                    return json.load(f)
            except:
                return {"svm": [], "random_forest": []}
        return {"svm": [], "random_forest": []}

    def _save_calibration(self):
        """Save calibration data to file with better error handling"""
        try:
            # Ensure directory exists before saving
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.calibration_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save calibration data: {str(e)}")

    def _get_system_load(self):
        """Get current system load factors with better error handling"""
        try:
            load_info = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
            }
            
            # Try to get CPU frequency, but don't fail if unavailable
            try:
                freq = psutil.cpu_freq()
                if freq:
                    load_info["cpu_freq"] = freq.current
            except Exception:
                load_info["cpu_freq"] = None
                
            return load_info
            
        except Exception as e:
            print(f"Warning: Error getting system load ({str(e)}). Using default values.")
            return {
                "cpu_percent": 50,  # Default to middle value
                "memory_percent": 50,
                "cpu_freq": None
            }

    def _get_cpu_count(self):
        """Get CPU count with fallback options"""
        try:
            return joblib_cpu_count()  # Try joblib first
        except:
            try:
                return mp_cpu_count()  # Fallback to multiprocessing
            except:
                return 1  # Final fallback
            
    def estimate_svm_time(self, n_samples, n_features):
        """Estimate SVM training time with calibration"""
        # Base estimation
        base_coefficient = 1e-7
        feature_coefficient = 1e-6
        
        # Calculate base estimate
        base_estimate = (
            base_coefficient * (n_samples ** 1.5) +
            feature_coefficient * n_features * np.log(n_samples)
        )

        # Adjust based on historical data if available
        if self.history["svm"]:
            recent_runs = self.history["svm"][-5:]  # Last 5 runs
            if recent_runs:
                adjustment_factors = [
                    run["actual_time"] / run["estimated_time"]
                    for run in recent_runs
                ]
                adjustment = np.median(adjustment_factors)
                base_estimate *= adjustment

        # Add system load factor
        load = self._get_system_load()
        load_factor = 1 + (load["cpu_percent"] / 100) * 0.5
        estimated_seconds = base_estimate * load_factor + 2.0  # 2s overhead

        # Store current estimation
        self.current_run["svm"] = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": n_samples,
            "n_features": n_features,
            "estimated_time": estimated_seconds,
            "system_load": load
        }

        return self._format_estimate(estimated_seconds)

    def estimate_rf_time(self, n_samples, n_features, n_trees, n_jobs):
        """Estimate Random Forest training time with robust system checks"""
        # Get CPU count and calculate parallel factor outside try block
        cpu_cores = self._get_cpu_count()
        n_jobs = abs(n_jobs) if n_jobs != 0 else cpu_cores
        parallel_factor = max(1, min(n_jobs, cpu_cores))
        
        try:
            # Get system load safely
            load = self._get_system_load()
            
            # Calculate base estimate
            base_time_per_tree = 0.001
            sample_factor = n_samples * np.log(n_samples)
            feature_factor = np.sqrt(n_features)
            
            # Calculate estimated time
            estimated_seconds = (
                (base_time_per_tree * n_trees * sample_factor * feature_factor) / 
                parallel_factor
            )
            
            # Apply load factor (only if available)
            load_factor = 1 + (load.get("cpu_percent", 50) / 100) * 0.3
            estimated_seconds *= load_factor
            
        except Exception as e:
            print(f"Warning: Error in RF time estimation ({str(e)}). Using simplified estimate.")
            # Fallback to simple estimation
            estimated_seconds = (n_samples * n_trees) / (1000 * parallel_factor)

        # Store estimation data (outside try block)
        self.current_run["random_forest"] = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": n_samples,
            "n_features": n_features,
            "n_trees": n_trees,
            "n_jobs": n_jobs,
            "estimated_time": estimated_seconds,
            "system_load": self._get_system_load(),
            "parallel_factor": parallel_factor  # Store for reference
        }
        
        return self._format_estimate(estimated_seconds)


    def record_actual_time(self, model_type, actual_time):
        """Record actual training time and update calibration"""
        if model_type.lower() not in self.current_run:
            return

        run_data = self.current_run[model_type.lower()]
        run_data["actual_time"] = actual_time
        self.history[model_type.lower()].append(run_data)
        
        # Keep only last 20 runs
        self.history[model_type.lower()] = self.history[model_type.lower()][-20:]
        
        self._save_calibration()
        self._print_accuracy_report(model_type.lower())

    def _format_estimate(self, seconds):
        """Format time estimate with uncertainty range"""
        def format_time(s):
            if s < 60:
                return f"{s:.1f} seconds"
            elif s < 3600:
                return f"{s/60:.1f} minutes"
            else:
                return f"{s/3600:.1f} hours"

        lower_bound = seconds * 0.7  # 30% lower
        upper_bound = seconds * 1.5  # 50% higher

        return {
            "estimate": seconds,
            "formatted": format_time(seconds),
            "range": {
                "low": format_time(lower_bound),
                "high": format_time(upper_bound)
            }
        }

    def _print_accuracy_report(self, model_type):
        """Print a report on estimation accuracy"""
        recent_runs = self.history[model_type][-5:]
        if not recent_runs:
            return

        print("\nTime Estimation Accuracy Report:")
        print(f"Model: {model_type.upper()}")
        print("\nRecent runs:")
        
        errors = []
        for run in recent_runs:
            error_ratio = run["actual_time"] / run["estimated_time"]
            errors.append(error_ratio)
            print(f"\nRun at {run['timestamp']}")
            print(f"Estimated: {run['estimated_time']:.2f}s")
            print(f"Actual: {run['actual_time']:.2f}s")
            print(f"Error ratio: {error_ratio:.2f}x")
            print(f"System load: CPU {run['system_load']['cpu_percent']}%")

        mae = np.mean(np.abs([e - 1 for e in errors]))
        print(f"\nMean Absolute Error: {mae:.2f}")
        print(f"Current calibration factor: {np.median(errors):.2f}")
