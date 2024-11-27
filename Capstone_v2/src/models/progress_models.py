import threading
from tqdm import tqdm
from sklearn.base import BaseEstimator
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

class ProgressMonitorMixin:
    """Base mixin for progress monitoring in ML models"""
    def _setup_progress_bar(self, total, desc, unit):
        return tqdm(
            total=total,
            desc=desc,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} " + unit + " "
                      "[{elapsed}<{remaining}, {rate_fmt}]"
        )
    
    def _initialize_progress(self):
        self.progress_lock = threading.Lock()
        self.completed = 0
    
    def _update_progress(self):
        with self.progress_lock:
            self.completed += 1
            if hasattr(self, 'pbar'):
                self.pbar.update(1)
    
    def _cleanup_progress(self):
        if hasattr(self, 'pbar'):
            self.pbar.close()

class ProgressRandomForest(RandomForestClassifier, ProgressMonitorMixin):
    def fit(self, X, y):
        self._initialize_progress()
        self.n_total_trees = self.n_estimators
        self.pbar = self._setup_progress_bar(
            total=self.n_estimators,
            desc="Building trees",
            unit="trees"
        )
        
        def update_progress(*args):
            self._update_progress()
        
        try:
            self.verbose = 0
            self._progress_callback = update_progress
            result = super().fit(X, y)
        finally:
            self._cleanup_progress()
        
        return result

class ProgressSVC(SVC, ProgressMonitorMixin):
    def fit(self, X, y):
        self._initialize_progress()
        n_samples = X.shape[0]
        estimated_iterations = int(n_samples * 1.2)
        
        self.pbar = self._setup_progress_bar(
            total=estimated_iterations,
            desc="Training SVM",
            unit="samples"
        )
        
        try:
            self.max_iter = estimated_iterations
            result = super().fit(X, y)
        finally:
            self._cleanup_progress()
        
        return result

class ProgressLinearSVC(LinearSVC, ProgressMonitorMixin):
    def fit(self, X, y):
        self._initialize_progress()
        self.pbar = self._setup_progress_bar(
            total=100,
            desc="Training LinearSVC",
            unit="%"
        )
        
        try:
            self.pbar.update(10)  # Initial setup
            result = super().fit(X, y)
            self.pbar.update(90)  # Complete progress
        finally:
            self._cleanup_progress()
        
        return result
