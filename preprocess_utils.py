import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def x_full_transform(x):
    return torch.Tensor(np.log(np.log(1/x)))

def x_state_transform(x):
    return torch.Tensor(np.sqrt(np.log(1e8*x)))

def x_heal_transform(x):
    return (torch.abs(torch.Tensor(x) - 1e-8) < 1e-12).float()

def prepare_data(model_type, state_only = False):

    if model_type == 'dual' and state_only:
        raise ValueError("Model cannot be dual and use only state data") 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not state_only:
        feature = np.genfromtxt('friction_data/features_AgingLaw_v2.csv', delimiter=',')[:, np.newaxis, :]
        target = np.genfromtxt('friction_data/targets_AgingLaw_v2.csv', delimiter=',')[:, np.newaxis, :]
    else:
        feature = np.genfromtxt('friction_data/features_AgingLaw_NoHealing.csv', delimiter=',')[:, np.newaxis, :]
        target = np.genfromtxt('friction_data/targets_AgingLaw_NoHealing.csv', delimiter=',')[:, np.newaxis, :]

    train_x = feature[:750, :, :]
    train_y = target[:750, :, :]
    test_x = feature[750:, :, :]
    test_y = target[750:, :, :]

    data = {"train_x": train_x,
            "train_y": train_y,
            "test_x": test_x,
            "test_y": test_y}

    if model_type == 'dual':
        train_x_norm_statePart = x_state_transform(train_x)
        x_max = train_x_norm_statePart.max()
        train_x_norm_statePart /= x_max
        train_x_norm_healPart = x_heal_transform(train_x)
        train_x_norm = torch.cat([train_x_norm_statePart, train_x_norm_healPart], dim=1)
    
        test_x_norm_state = x_state_transform(test_x) / x_max
        test_x_norm_heal = x_heal_transform(test_y)
        test_x_norm = torch.cat([test_x_norm_state, test_x_norm_heal], dim=1)

        feature_state = np.genfromtxt('friction_data/features_AgingLaw_NoHealing.csv', delimiter=',')[:, np.newaxis, :]
        target_state = np.genfromtxt('friction_data/targets_AgingLaw_NoHealing.csv', delimiter=',')[:, np.newaxis, :]

        train_x_state = feature_state[:750, :, :]
        train_y_state = target_state[:750, :, :]
        test_x_state = feature_state[750:, :, :]
        test_y_state = target_state[750:, :, :]

        train_x_norm_state = x_state_transform(train_x_state) / x_max
        test_x_norm_state = x_state_transform(test_x_state) / x_max

        train_y_norm_state = torch.Tensor(train_y_state / train_y_state.max())
        test_y_norm_state = torch.Tensor(test_y_state / train_y_state.max())
        train_dataset_state = TensorDataset(train_x_norm_state.to(device), train_y_norm_state.to(device))
        train_loader_state = DataLoader(train_dataset_state, batch_size=128, shuffle=True)

        data["train_y_norm_state"] = train_y_norm_state.to(device)
        data["test_y_norm_state"] = test_y_norm_state.to(device)
        data["train_x_norm_state"] = train_x_norm_state.to(device)
        data["test_x_norm_state"] = test_x_norm_state.to(device)
        data["train_dataset_state"] = train_dataset_state
        data["train_loader_state"] = train_loader_state

    else:
        train_x_norm = x_state_transform(train_x)
        test_x_norm = x_state_transform(test_x)

    train_y_norm = torch.Tensor(train_y / train_y.max())
    test_y_norm = torch.Tensor(test_y / train_y.max())
    
    train_dataset = TensorDataset(train_x_norm.to(device), train_y_norm.to(device))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    data["train_x_norm"] = train_x_norm.to(device)
    data['test_x_norm'] = test_x_norm.to(device)
    data['train_y_norm'] = train_y_norm.to(device)
    data['test_y_norm'] = test_y_norm.to(device)
    data['train_dataset'] = train_dataset
    data['train_loader'] = train_loader

    return data, device

def combine_with_suffix(heal_params, state_params):
    heal_with_suffix = {f"{k}_heal": v for k, v in heal_params.items()}
    state_with_suffix = {f"{k}_state": v for k, v in state_params.items()}
    return {**heal_with_suffix, **state_with_suffix}
