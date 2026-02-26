"""
Configuration for baseline Faster R-CNN
"""
from .base_config import BaseConfig


class BaselineConfig(BaseConfig):
    """Configuration for baseline model"""
    
    output_dir = 'outputs'
    enhancement_type = None
