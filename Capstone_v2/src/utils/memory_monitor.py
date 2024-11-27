import psutil
from humanize import naturalsize

class MemoryMonitor:
    @staticmethod
    def get_current_usage():
        process = psutil.Process()
        return naturalsize(process.memory_info().rss)
    
    @staticmethod
    def estimate_memory_requirement(n_samples, n_features, model_type):
        if model_type == "SVM":
            kernel_matrix = n_samples * n_samples * 8  # 8 bytes per float64
            support_vectors = n_samples * n_features * 8 * 0.3  # Assume 30% support vectors
            overhead = 1024 * 1024 * 100  # 100MB overhead
            return naturalsize(kernel_matrix + support_vectors + overhead)
        elif model_type == "Random Forest":
            tree_size = n_samples * n_features * 8 * 0.1  # Rough estimate per tree
            return naturalsize(tree_size)
        return "Unknown"
