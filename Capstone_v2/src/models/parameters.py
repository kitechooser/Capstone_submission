class ModelParameters:
    """Class to handle model parameters and search spaces - essentially just a shell around config.py now."""
    
    @staticmethod
    def get_default_params():
        """Get default parameters for each model - Deprecated, use config.get_default_params instead"""
        from config.config import Config
        config = Config()
        return {model: config.get_default_params(model) for model in config.ACTIVE_MODELS.keys()}

    @staticmethod
    def get_search_spaces():
        """Get parameter search spaces for each model - Deprecated, use config.get_search_space instead"""
        from config.config import Config
        config = Config()
        return {model: config.get_search_space(model) for model in config.ACTIVE_MODELS.keys()}
