from models.architectures.cnn import create_cnn
from models.architectures.mesonet import create_mesonet

class ModelFactory:
    """Factory class to create different model architectures"""
    
    def __init__(self, config):
        self.config = config
    
    def create_model(self, model_name, params=None):
        """Create a model based on name and parameters"""
        try:
            if params is None:
                params = {}
                
            if model_name == "CNN":
                return create_cnn(self.config, **params)
            elif model_name == "Mesonet":
                return create_mesonet(self.config, **params)
            elif model_name == "SVM":
                from models.progress_models import ProgressSVC
                return ProgressSVC(**params.get("model_params", {}))
            elif model_name == "Random Forest":
                from models.progress_models import ProgressRandomForest
                return ProgressRandomForest(**params.get("model_params", {}))
            else:
                raise ValueError(f"Unknown model type: {model_name}")
                
        except Exception as e:
            print(f"Error creating model {model_name}: {str(e)}")
            print(f"Params: {params}")
            raise e
