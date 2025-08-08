import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import warnings

def x_log_transform(x):
    return torch.Tensor(np.log(np.log(1/x)))

def x_state_transform(x):
    return torch.Tensor(np.sqrt(np.log(1e8*x)))

def x_heal_transform(x):
    return (torch.abs(torch.Tensor(x) - 1e-8) < 1e-12).float()

def prepare_data(config, device):
    print("PREPARING DATA")

    feature = np.genfromtxt(config['data']['x_path'], delimiter=',')[:, np.newaxis, :]
    target = np.genfromtxt(config['data']['y_path'], delimiter=',')[:, np.newaxis, :]

    train_x = feature[:config['data']['train_samples'], :, :]
    train_y = target[:config['data']['train_samples'], :, :]
    test_x = feature[config['data']['train_samples']:, :, :]
    test_y = target[config['data']['train_samples']:, :, :]

    data = {"train_x": train_x,
            "train_y": train_y,
            "test_x": test_x,
            "test_y": test_y}
    
    train_norm_components = []
    test_norm_components = []
    
    if config['data']['log_norm']:
        print("\t USING LOG NORM DATA")
        train_x_norm_log = x_log_transform(train_x)
        x_max = train_x_norm_log.max()
        train_x_norm_log /= x_max
        test_x_norm_log = x_log_transform(test_x) / x_max
        
        train_norm_components.append(train_x_norm_log)
        test_norm_components.append(test_x_norm_log)
        data['train_x_norm_log'] = train_x_norm_log
        data['test_x_norm_log'] = test_x_norm_log
    
    if config['data']['state_norm']:
        print("\t USING STATE NORM DATA")
        train_x_norm_state = x_state_transform(train_x)
        x_max = train_x_norm_state.max()
        train_x_norm_state /= x_max 
        test_x_norm_state = x_state_transform(test_x) / x_max
                
        train_norm_components.append(train_x_norm_state)
        test_norm_components.append(test_x_norm_state)
        data['train_x_norm_state'] = train_x_norm_state
        data['test_x_norm_state'] = test_x_norm_state


    if config['data']['heal_norm']:
        print("\t USING HEAL NORM DATA")
        train_x_norm_heal = x_heal_transform(train_x)
        test_x_norm_heal = x_heal_transform(test_x)

        train_norm_components.append(train_x_norm_heal)
        test_norm_components.append(test_x_norm_heal)
        data['train_x_norm_heal'] = train_x_norm_state
        data['test_x_norm_heal'] = test_x_norm_state

    if 'pretrain' in config:
        feature_SO = np.genfromtxt("data/friction_data/features_AgingLaw_NoHealing.csv", delimiter=',')[:, np.newaxis, :]
        target_SO = np.genfromtxt("data/friction_data/targets_AgingLaw_NoHealing.csv", delimiter=',')[:, np.newaxis, :]

        train_x_SO = feature_SO[:config['data']['train_samples'], :, :]
        train_y_SO = target_SO[:config['data']['train_samples'], :, :]
        test_x_SO = feature_SO[config['data']['train_samples']:, :, :]
        test_y_SO = target_SO[config['data']['train_samples']:, :, :]

        train_x_norm_SO = x_state_transform(train_x_SO)
        x_max = train_x_norm_SO.max()
        train_x_norm_SO /= x_max 
        test_x_norm_SO = x_state_transform(test_x_SO) / x_max

        train_y_norm_SO = torch.Tensor(train_y_SO / train_y_SO.max())
        test_y_norm_SO = torch.Tensor(test_y_SO / train_y_SO.max())

        train_dataset_SO = TensorDataset(train_x_norm_SO.to(device), train_y_norm_SO.to(device))
        train_loader_SO = DataLoader(train_dataset_SO, batch_size=32, shuffle=True)
                
        data['train_x_norm_SO'] = train_x_norm_SO.to(device)
        data['test_x_norm_SO'] = test_x_norm_SO.to(device)
        data['train_y_norm_SO'] = train_y_norm_SO.to(device)
        data['test_y_norm_SO'] = test_y_norm_SO.to(device)
        data['train_dataset_SO'] = train_dataset_SO
        data['train_loader_SO'] = train_loader_SO

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


def check_lift_or_decode(name, block):
    if "NN" not in block:
        raise KeyError(f"Missing key 'NN' in {name} config")
    
    if block["NN"]:
        if "NN_params" not in block:
            raise KeyError(f"Missing 'NN_params' in {name} config while 'NN' is True")
        for key in ["width", "depth"]:
            if key not in block["NN_params"]:
                raise KeyError(f"Missing key '{key}' in {name}['NN_params']")
    else:
        if "NN_params" in block:
            raise KeyError(f"Unexpected 'NN_params' in {name} config while 'NN' is False")

def check_fno_block(name, fno_block):
    required_keys = ["mode", "blocks", "act", "width", "padding", "coord_features", "adaptive"]
    missing = [k for k in required_keys if k not in fno_block]
    if missing:
        raise KeyError(f"Missing keys in {name}['fno']: {missing}")

def check_component(name, comp):
    if "fno" not in comp:
        raise KeyError(f"Missing 'fno' in {name} config")
    if "lift" not in comp:
        raise KeyError(f"Missing 'lift' in {name} config")
    if "decode" not in comp:
        raise KeyError(f"Missing 'decode' in {name} config")

    check_fno_block(f"{name}", comp["fno"])
    check_lift_or_decode(f"{name}['lift']", comp["lift"])
    check_lift_or_decode(f"{name}['decode']", comp["decode"])

def check_config(config):
    # --- Data ---
    if 'data' not in config:
        raise KeyError("Missing key 'data' used to prepare training and test data")
    else:
        required_data_keys = ['x_path', 'y_path', 'train_samples', 'log_norm', 'state_norm', 'heal_norm']
        missing_data = [k for k in required_data_keys if k not in config['data']]
        if missing_data:
            raise KeyError(f"Missing keys in data_config: {missing_data}")

        if not (config['data']['log_norm'] or config['data']['state_norm'] or config['data']['heal_norm']):
            raise KeyError("At least one type of data normalization must be chosen")

        if config['model_type'] == 'dual':
            if config['data']['log_norm'] or not (config['data']['state_norm'] and config['data']['heal_norm']):
                raise KeyError("For a dual model, both state_norm and heal_norm should be used, without log_norm")

    # --- Training ---
    if 'training' not in config:
        raise KeyError("Missing key 'training' used to setup training loop")
    else:
        required_train_keys = ['lr', 'save_results', 'epochs']
        missing_train = [k for k in required_train_keys if k not in config['train']]
        if missing_train:
            raise KeyError(f"Missing keys in train_config: {missing_train}")

    # --- Model ---
    if config['model_type'] == 'single':
        if 'pretrain' in config:
            warnings.warn("WARNING: Cannot use pretraining for single FNO model, will simply be ignored")

        check_component("single", config)

    elif config['model_type'] == 'dual':
        # --- Pretrain ---
        if 'pretrain' in config:
            required_pretrain_keys = ["load", "lr", "epochs", "save_results"]
            missing_pretrain = [k for k in required_pretrain_keys if k not in config["pretrain"]]
            if missing_pretrain:
                raise KeyError(f"Missing keys in pretrain config: {missing_pretrain}")

        # --- Heal and State blocks ---
        if 'heal' not in config:
            raise KeyError("Missing 'heal' component in dual model config")
        if 'state' not in config:
            raise KeyError("Missing 'state' component in dual model config")

        check_component("heal", config["heal"])
        check_component("state", config["state"])
    else:
        raise ValueError("Invalid model_type, must be 'single' or 'dual'")
