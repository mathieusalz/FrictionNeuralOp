from copy import deepcopy as dc
import torch
import os
import pprint
from FNO import FNO1d

def save_config_as_text(config: dict, path: str) -> None:
    """
    Save a configuration dictionary as a Python-formatted text file named 'config.txt' in the given directory path,
    so that the user can directly copy-paste the config dictionary into a script.

    Args:
        config (dict): Configuration dictionary to save.
        path (str): Directory path where 'config.txt' will be saved.
    """
    
    # Deepcopy to avoid modifying original config
    config_copy = dc(config)

    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)

    full_path = os.path.join(path, "config.txt")
    with open(full_path, 'w') as f:
        f.write("config = ")
        f.write(pprint.pformat(config_copy))
        f.write("\n")

def save_model(model: torch.nn.Module, name: str = 'model') -> None:
    """
    Save the model's state dictionary and all relevant initialization parameters to a file.

    Args:
        model (torch.nn.Module): The model to save.
        name (str, optional): Base filename for saving the model. Defaults to 'model'.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_params': {
            'in_channels': model.in_channels - (1 if model.coord_features else 0),  # undo coord feature adjustment
            'out_channels': model.out_channels,
            'modes': model.modes,
            'width': model.width,
            'block_activation': model.block_activation,
            'n_blocks': model.n_blocks,
            'padding': model.padding,
            'coord_features': model.coord_features,
            'adaptive': model.adaptive,
            'lift_activation': model.lift_activation,
            'lift_NN': model.lift_NN,
            'lift_NN_params': model.lift_NN_params if model.lift_NN else None,
            'decode_activation': model.decode_activation,
            'decode_NN': getattr(model, 'decode_NN', False),
            'decode_NN_params': getattr(model, 'decode_NN_params', None) if getattr(model, 'decode_NN', False) else None
        }
    }, f'{name}.pth')



def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    """
    Load a saved model from a file, reconstruct it with saved parameters, and load its state dictionary.

    Args:
        model_name (str): Base filename (without extension) of the saved model.
        device (torch.device): Device to map the loaded model to.

    Returns:
        torch.nn.Module: The loaded model instance.
    """    

    checkpoint = torch.load(f'{model_name}.pth', map_location=device, weights_only=False)
    model_params = checkpoint['model_params']
    model = FNO1d(**model_params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model