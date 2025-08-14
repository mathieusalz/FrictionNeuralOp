import torch
import torch.nn as nn
import torch.optim as optim
from FNO import FNO1d, dual_FNO
from save_utils import load_model
import torch.nn.utils as utils
from typing import Tuple
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

def get_activation(act_name: str) -> nn.Module:
    """
    Map a string to a PyTorch activation module.
    
    Args:
        act_name (str): Name of the activation function.
    
    Returns:
        nn.Module: Corresponding activation function module.
    
    Raises:
        ValueError: If the activation name is unknown.
    """
    act_name = act_name.lower()
    if act_name == 'gelu':
        return nn.GELU()
    elif act_name == 'silu':
        return nn.SiLU()
    elif act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'leakyrelu':
        return nn.LeakyReLU()
    elif act_name == 'tanh':
        return nn.Tanh()
    elif act_name == 'none':
        return nn.Identity()
    # Add more activations as needed
    else:
        raise ValueError(f"Unknown activation function '{act_name}' specified in config.")


def model_setup(config : dict, 
                data: dict, 
                device: torch.device) -> nn.Module:
    """
    Initializes and returns a model based on the provided configuration and data.

    For a 'single' model_type, sets up a single FNO1d model with parameters from config.
    For a 'dual' model_type, sets up separate FNO1d models for 'heal' and 'state' and combines
    them into a dual_FNO model.

    Args:
        config (dict): Configuration dictionary specifying model parameters and type.
        data (dict): Dictionary containing training data tensors for shape inference.
        device (torch.device): Device to place the model on (CPU or GPU).

    Returns:
        nn.Module: Instantiated model ready for training.
    """

    if config['model_type'] == 'single':
        in_channels = data['train_x_norm'].shape[1]
        out_channels = data['train_y_norm'].shape[1]
        model = FNO1d(
                in_channels       = in_channels,
                out_channels      = out_channels,
                modes             = config['fno']['mode'],
                width             = config['fno']['width'],
                block_activation  = get_activation(config['fno']['act']),
                n_blocks          = config['fno']['blocks'],
                padding           = config['fno']['padding'],
                coord_features    = config['fno']['coord_features'],
                adaptive          = config['fno']['adaptive'],
                lift_activation   = get_activation(config['lift']['act']),
                lift_NN           = config['lift']['NN'],
                lift_NN_params    = config['lift']['NN_params'] if 'NN_params' in config['lift'] else None,
                decode_activation = get_activation(config['decode']['act']),
                decode_NN         = config['decode']['NN'],
                decode_NN_params  = config['decode']['NN_params'] if 'NN_params' in config['decode'] else None,
                ).to(device)
    
    else:
        fno_heal = FNO1d(
                in_channels       = 1,
                out_channels      = 1,
                modes             = config['heal']['fno']['mode'],
                width             = config['heal']['fno']['width'],
                block_activation  = get_activation(config['heal']['fno']['act']),
                n_blocks          = config['heal']['fno']['blocks'],
                padding           = config['heal']['fno']['padding'],
                coord_features    = config['heal']['fno']['coord_features'],
                adaptive          = config['heal']['fno']['adaptive'],
                lift_activation   = get_activation(config['heal']['lift']['act']),
                lift_NN           = config['heal']['lift']['NN'],
                lift_NN_params    = config['heal']['lift']['NN_params'] if 'NN_params' in config['heal']['lift'] else {},
                decode_activation = get_activation(config['heal']['decode']['act']),
                decode_NN         = config['heal']['decode']['NN'],
                decode_NN_params  = config['heal']['decode']['NN_params'] if 'NN_params' in config['heal']['decode'] else {},
                ).to(device)

        fno_state = FNO1d(
                in_channels       = 1,
                out_channels      = 1,
                modes             = config['state']['fno']['mode'],
                width             = config['state']['fno']['width'],
                block_activation  = get_activation(config['state']['fno']['act']),
                n_blocks          = config['state']['fno']['blocks'],
                padding           = config['state']['fno']['padding'],
                coord_features    = config['state']['fno']['coord_features'],
                adaptive          = config['state']['fno']['adaptive'],
                lift_activation   = get_activation(config['state']['lift']['act']),
                lift_NN           = config['state']['lift']['NN'],
                lift_NN_params    = config['state']['lift']['NN_params'] if 'NN_params' in config['state']['lift'] else {},
                decode_activation = get_activation(config['state']['decode']['act']),
                decode_NN         = config['state']['decode']['NN'],
                decode_NN_params  = config['state']['decode']['NN_params'] if 'NN_params' in config['state']['decode'] else {},
                ).to(device)

        model = dual_FNO(fno_heal, fno_state).to(device)

    return model

def make_decay_fn(factor: float, 
                  interval: int):
    """
    Creates a learning rate decay function for a LambdaLR scheduler.

    The returned function decays the learning rate by 'factor' every 'interval' steps.

    Args:
        factor (float): Multiplicative decay factor (e.g., 0.9).
        interval (int): Number of steps between each decay.

    Returns:
        Callable[[int], float]: Function mapping the current step to the decay multiplier.
    """
    return lambda step: factor ** (step // interval)

def train_loop(model: nn.Module,
               train_loader: DataLoader,
               val_x: torch.Tensor,
               val_y: torch.Tensor,
               epochs: int,
               optimizer: Optimizer,
               scheduler: _LRScheduler,
               criterion: nn.modules.loss._Loss = nn.MSELoss(),
               save_results: bool = True,
               clip_grad: bool = False,
               add_noise: bool = False,
               verbose: bool = False,
               sample_freq: int = 1) -> Tuple[list, list]:
    """
    Runs the training loop for a specified number of epochs.

    For each epoch, trains the model on the training data with optional noise and downsampling,
    applies gradient clipping, updates optimizer and scheduler, and evaluates on validation data.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader providing training batches.
        val_x (torch.Tensor): Validation input data.
        val_y (torch.Tensor): Validation target data.
        optimizer (Optimizer): Optimizer instance to update model parameters.
        scheduler (_LRScheduler): Learning rate scheduler.
        epochs (int): Number of epochs to train.
        criterion (nn.modules.loss._Loss, optional): Loss function. Defaults to MSELoss.
        save_results (bool, optional): Whether to save loss history each epoch. Defaults to True.

    Returns:
        Tuple[list, list]: Tuple containing training loss history and validation loss history.
    """
    
    loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            
            if add_noise:
                rand_int = [1,2,3,4][torch.multinomial(torch.Tensor([0.6,0.25,0.10,0.05]), 1).item()]
                batch_x_down = batch_x[:,:,::rand_int]
                
                std = torch.rand(1).item() * 0.015
                noise = torch.randn_like(batch_x_down) * std
                batch_x_noisy = batch_x_down + noise
                output = model(batch_x_noisy)
                loss = criterion(output, batch_y[:,:,::rand_int])
            else:
                output = model(batch_x[:,:,::sample_freq])
                loss = criterion(output, batch_y[:,:,::sample_freq])

            loss.backward()

            if clip_grad:
                utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
            epoch_loss += loss.item()
            scheduler.step()

        if save_results or epoch == epochs -1:
            loss_history.append(epoch_loss / len(train_loader))
            
            model.eval()
            with torch.no_grad():
                val_output = model(val_x)
                val_loss = criterion(val_output, val_y).item()
                val_loss_history.append(val_loss)

            if verbose:
                print(f"EPOCH {epoch}: train_loss: {loss.item():.3e} \t test_loss: {val_loss:.3e}")

    return loss_history, val_loss_history

def pretraining(model: nn.Module, 
                data: dict, 
                device: torch.device, 
                config: dict, 
                criterion: nn.modules.loss._Loss) -> None:
    """
    Handles optional pretraining of the state component of a dual model.

    If pretraining is enabled and load is True, loads a pretrained model.
    Otherwise, trains the state model using the provided pretraining dataset and configuration.

    Args:
        model (nn.Module): The dual model containing a FNO_state submodel.
        data (dict): Dataset dictionary including pretraining loaders.
        device (torch.device): Device to run training on.
        config (dict): Configuration dictionary with pretraining parameters.
        criterion (nn.modules.loss._Loss): Loss function to optimize.

    Returns:
        None
    """
    
    if config['pretrain']['load'] == True:
        print("\t LOADING PRETRAINED")
        fno_state = load_model("state_model", device)
        model.FNO_state = fno_state

    else:
        print(f'\t PRETRAINING for {config["pretrain"]["epochs"]}')
        train_loader_state = data['train_loader_SO']
        test_x_state = data['test_x_norm_SO']
        test_y_state = data['test_y_norm_SO']

        optimizer_state = optim.Adam(model.FNO_state.parameters(), lr=config['pretrain']['lr'])
        scheduler_state = torch.optim.lr_scheduler.LambdaLR(optimizer_state, lr_lambda=make_decay_fn(0.9, 1000))

        loss_history_state, val_loss_history_state = train_loop(model.FNO_state, train_loader_state, test_x_state, test_y_state,
                                                                optimizer_state, scheduler_state, config['pretrain']['epochs'], 
                                                                criterion, config['pretrain']['save_results'])

        data['loss_history_state'] = loss_history_state
        data['val_loss_history_state'] = val_loss_history_state
        

def train_model(config: dict, 
                model: nn.Module, 
                data: dict, device: 
                torch.device):
    """
    Orchestrates the full training procedure based on the configuration.

    For dual models, optionally performs pretraining on the state submodel, sets up
    separate optimizers with learning rate factors, and disables training if factor is near zero.
    For single models, trains the entire model with a single optimizer.

    Updates the data dictionary with training and validation loss histories.

    Args:
        config (dict): Configuration dictionary with training parameters.
        model (nn.Module): The model to train.
        data (dict): Dataset and preprocessed data.
        device (torch.device): Device to run training on.

    Returns:
        None
    """
    
    print(f"Beginning Training for {config['train']['epochs']}")
    criterion = nn.MSELoss()
    lr = config['train']["lr"]

    if config['model_type'] == 'dual':

        if 'pretrain' in config:
            pretraining(model, data, device, config, criterion)

        lr_state_factor = config['train']["lr_state_factor"] if "lr_state_factor" in config['train'] else 1

        optimizer = optim.Adam([
            {"params": model.FNO_state.parameters(), "lr": lr * lr_state_factor, "weight_decay": 1e-4},
            {"params": model.FNO_heal.parameters(), "lr": lr, "weight_decay": 1e-4}
        ])

        if lr_state_factor < 1e-6:
            for param in model.FNO_state.parameters():
                param.requires_grad = False

    else: 
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=make_decay_fn(data['train']['decay'], data['train']['steps']))

    loss_history, val_loss_history = [], []
    train_loader, test_x, test_y = data['train_loader'], data['test_x_norm'], data['test_y_norm']

    loss_history, val_loss_history = train_loop(model, train_loader, test_x, test_y,
                                                optimizer, scheduler, config['train']['epochs'], 
                                                criterion, config['train']['save_results'])

    data['loss_history'] = loss_history
    data['val_loss_history'] = val_loss_history
    print("\t End of Training")