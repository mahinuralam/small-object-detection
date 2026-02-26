"""
Configuration for DPA-enhanced training
"""
from .base_config import BaseConfig


class DPAConfig(BaseConfig):
    """Configuration for DPA-enhanced model"""
    
    # Override batch size for DPA (needs more memory)
    batch_size = 4
    
    # Output directory
    output_dir = 'outputs_dpa'
    
    # Enhancement settings
    enhancement_type = 'dpa'
    fpn_channels = 256
    enhance_levels = ['0', '1']  # P3, P4
    
    # DPA specific
    use_spatial_attention = True
    use_channel_attention = True
    multi_scale_kernels = [3, 5]
