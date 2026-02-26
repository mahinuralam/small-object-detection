"""
Base configuration for all experiments
"""

class BaseConfig:
    """Base configuration class"""
    
    # Dataset
    dataset_root = '../dataset/VisDrone-2018'
    num_classes = 10
    min_object_size = 5
    
    # Training
    batch_size = 4
    num_workers = 4
    num_epochs = 50
    learning_rate = 0.005
    momentum = 0.9
    weight_decay = 0.0005
    
    # Learning rate schedule
    lr_step_size = 10
    lr_gamma = 0.1
    
    # Early stopping
    patience = 10
    save_interval = 5
    
    # Device
    device = 'cuda'
    
    # Model
    backbone = 'resnet50'
    pretrained = True
    trainable_backbone_layers = 3
    
    # Output
    output_dir = 'outputs'
    
    def __repr__(self):
        config_str = "Configuration:\n"
        for key, value in self.__class__.__dict__.items():
            if not key.startswith('_') and not callable(value):
                config_str += f"  {key}: {value}\n"
        return config_str
