import importlib
from pathlib import Path
import sys

def load_model(model_name):
    """
    Load a model class by name.
    
    Args:
        model_name (str): Name of the model to load
    
    Returns:
        model_class: The requested model class
    """
    try:
        if model_name == 'DGDNN':
            from models.DGDNN.Model.dgdnn import DGDNN
            return DGDNN
        elif model_name == 'GraphWaveNet':
            from models.GraphWaveNet.gwnet import gwnet
            return gwnet
        elif model_name == 'DARNN':
            from models.DARNN.DARNN import DARNN
            return DARNN
        elif model_name == 'HyperStockGAT':
            from models.HyperStockGAT.training.models.base_models import NCModel
            return NCModel
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except ImportError as e:
        print(f"Error loading model {model_name}: {str(e)}")
        raise
