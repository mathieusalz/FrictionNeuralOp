import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def x_full_transform(x):
    return torch.Tensor(np.log(np.log(1/x)))

def x_state_transform(x):
    return torch.Tensor(np.sqrt(np.log(1e8*x)))

def x_heal_transform(x):
    return (torch.abs(torch.Tensor(x) - 1e-8) < 1e-12).float()

def prepare_data(data_config, device):

    feature = np.genfromtxt(data_config['x_path'], delimiter=',')[:, np.newaxis, :]
    target = np.genfromtxt(data_config['y_path'], delimiter=',')[:, np.newaxis, :]

    train_x = feature[:data_config['train_samples'], :, :]
    train_y = target[:data_config['train_samples'], :, :]
    test_x = feature[data_config['train_samples']:, :, :]
    test_y = target[data_config['train_samples']:, :, :]

    data = {"train_x": train_x,
            "train_y": train_y,
            "test_x": test_x,
            "test_y": test_y}
    
    train_norm_components = []
    test_norm_components = []
    
    if data_config['log_norm']:
        train_x_norm_full = x_full_transform(train_x)
        x_max = train_x_norm_full.max()
        train_x_norm_log /= x_max
        test_x_norm_log = x_full_transform(test_x) / x_max
        train_norm_components.append(train_x_norm_log)
        test_norm_components.append(test_x_norm_log)
        data['train_x_norm_log'] = train_x_norm_log
        data['test_x_norm_log'] = test_x_norm_log
    
    if data_config['state_norm']:
        train_x_norm_state = x_state_transform(train_x)
        x_max = train_x_norm_state.max()
        train_x_norm_state /= x_max 
        test_x_norm_state = x_state_transform(test_x) / x_max
        train_norm_components.append(train_x_norm_state)
        test_norm_components.append(test_x_norm_state)
        data['train_x_norm_state'] = train_x_norm_state
        data['test_x_norm_state'] = test_x_norm_state


    if data_config['heal_norm']:
        train_x_norm_heal = x_heal_transform(train_x)
        test_x_norm_heal = x_heal_transform(test_x)
        train_norm_components.append(train_x_norm_heal)
        test_norm_components.append(test_x_norm_heal)
        data['train_x_norm_heal'] = train_x_norm_state
        data['test_x_norm_heal'] = test_x_norm_state

    train_x_norm = torch.cat(train_norm_components, dim=1)
    test_x_norm = torch.cat(test_norm_components, dim=1)
     
    train_y_norm = torch.Tensor(train_y / train_y.max())
    test_y_norm = torch.Tensor(test_y / train_y.max())
    
    train_dataset = TensorDataset(train_x_norm.to(device), train_y_norm.to(device))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    data["train_x_norm"] = train_x_norm.to(device)
    data['test_x_norm'] = test_x_norm.to(device)
    data['train_y_norm'] = train_y_norm.to(device)
    data['test_y_norm'] = test_y_norm.to(device)
    data['train_dataset'] = train_dataset
    data['train_loader'] = train_loader

    return data

def combine_with_suffix(heal_params, state_params):
    heal_with_suffix = {f"{k}_heal": v for k, v in heal_params.items()}
    state_with_suffix = {f"{k}_state": v for k, v in state_params.items()}
    return {**heal_with_suffix, **state_with_suffix}
