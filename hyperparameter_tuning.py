import optuna
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from train_utils import train_model, prepare_data


def objective_dual(trial):
    config = {
    "model_type": 'dual',
    "width_NN_heal":  trial.suggest_int("width_NN_heal", 8, 512, log=True),
    "width_NN_state": trial.suggest_int("width_NN_state", 8, 512, log=True),
    "depth_NN_heal":  trial.suggest_int("depth_NN_heal", 1, 8),
    "depth_NN_state": trial.suggest_int("depth_NN_state", 1, 8),
    "mode_heal":      trial.suggest_int("mode_heal", 12, 64, step=4),
    "mode_state":     trial.suggest_int("mode_state", 12, 64, step=4),
    "blocks_heal":    trial.suggest_int("blocks_heal", 1, 8, step=1),
    "blocks_state":   trial.suggest_int("blocks_state", 1, 8, step=1),
    "lift_act_state": trial.suggest_categorical(
        "lift_act_state", [None, torch.nn.functional.tanh, torch.nn.functional.gelu]),
    "block_act_state": trial.suggest_categorical(
        "block_act_state", [None, torch.nn.functional.tanh, torch.nn.functional.gelu]),
    "lift_act_heal": trial.suggest_categorical(
        "lift_act_heal", [None, torch.nn.functional.tanh, torch.nn.functional.gelu]),
    "block_act_heal": trial.suggest_categorical(
        "block_act_heal", [None, torch.nn.functional.tanh, torch.nn.functional.gelu]),
    "width_heal":     trial.suggest_int("width_heal", 4, 128, log=True),
    "width_state":    trial.suggest_int("width_state", 4, 128, log=True),
    "lr":             trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    "padding_heal":   trial.suggest_int("padding_heal", 0, 32, step=1),
    "padding_state":  trial.suggest_int("padding_state", 0, 32, step=1),
    "optimizer_type": trial.suggest_categorical("optimizer_type", ["adam", "lbfgs"]),}


    _, _, val_loss = train_model(config, train_loader, test_x, test_y, device)
    return val_loss

def objective_single(trial):
    config = {
    "model_type": "single",
    "width_NN":  trial.suggest_int("width_NN", 8, 512, log=True),
    "depth_NN":  trial.suggest_int("depth_NN", 1, 8),
    "modes":      trial.suggest_int("modes", 12, 64, step=4),
    "n_blocks":    trial.suggest_int("blocks", 1, 8, step=1),
    "block_act": trial.suggest_categorical(
        "block_act", [None, torch.nn.functional.tanh, torch.nn.functional.gelu]),
    "lift_act": trial.suggest_categorical(
        "lift_act", [None, torch.nn.functional.tanh, torch.nn.functional.gelu]),
    "width":     trial.suggest_int("width", 4, 128, log=True),
    "lr":             trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    "padding":  trial.suggest_int("padding", 0, 32, step=1)}


    _, _, val_loss = train_model(config, train_loader, test_x, test_y, device)
    return val_loss[-1]


if __name__ == "__main__":
    train_loader, _, _, test_x, test_y, device = prepare_data('single')

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_single(trial), n_trials=1500)

    print("Best trial:")
    print(study.best_trial)
