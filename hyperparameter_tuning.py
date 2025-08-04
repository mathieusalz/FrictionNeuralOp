import optuna
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from train_utils import train_model
from preprocess_utils import prepare_data, combine_with_suffix


def objective_dual(trial,data):
    training_settings = {"model_type": 'dual',
                         "pretrain": False,
                         "pretrain_epochs": None,
                         "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                         "lr_state_factor": 1,
                         }
    
    state_config = {"width_NN":  trial.suggest_categorical("width_NN_state", [8, 64, 256, 512]),
                    "depth_NN":  trial.suggest_categorical("depth_NN_state", [1,4,8]),
                    "mode":      trial.suggest_categorical("modes_state", [12,32,64]),
                    "blocks":    trial.suggest_int("blocks_state", 1, 8, step=1),   
                    "lift_act": trial.suggest_categorical(
                        "lift_act_state", [None, torch.nn.functional.tanh, 
                                        torch.nn.functional.gelu, torch.nn.functional.mish]),
                    "block_act": trial.suggest_categorical(
                        "block_act_state", [None, torch.nn.functional.tanh, 
                                        torch.nn.functional.gelu, torch.nn.functional.mish]),
                    "width":     trial.suggest_categorical("width_state", [1,8,32,64]),
                    "padding":   trial.suggest_categorical("padding_state", [0,9,18]),
                    }
    
    heal_config = {"width_NN":  trial.suggest_categorical("width_NN_heal", [8, 64, 256, 512]),
                    "depth_NN":  trial.suggest_categorical("depth_NN_heal", [1,4,8]),
                    "mode":      trial.suggest_categorical("modes_heal", [12,32,64]),
                    "blocks":    trial.suggest_int("blocks_heal", 1, 8, step=1),   
                    "lift_act": trial.suggest_categorical(
                        "lift_act_heal", [None, torch.nn.functional.tanh, 
                                        torch.nn.functional.gelu, torch.nn.functional.mish]),
                    "block_act": trial.suggest_categorical(
                        "block_act_heal", [None, torch.nn.functional.tanh, 
                                        torch.nn.functional.gelu, torch.nn.functional.mish]),
                    "width":     trial.suggest_categorical("width_heal", [1,8,32,64]),
                    "padding":   trial.suggest_categorical("padding_heal", [0,9,18]),
                    }

    config = {**training_settings, **combine_with_suffix(heal_config, state_config)}

    _, data_out = train_model(config, data, device)
    val_loss = data_out["val_loss_history"][-1]
    train_loss = data_out["loss_history"][-1]
    print(f"[Trial {trial.number}] Train loss: {train_loss:.3e}, Val loss: {val_loss:.3e}, Overfit gap: {val_loss - train_loss:.3e}", flush = True)

    # Optional overfitting penalty (e.g. L2 gap penalty)
    penalty_lambda = 1.0  # You can tune this
    objective_value = val_loss + penalty_lambda * max(val_loss - train_loss, 0)

    return objective_value

def objective_single(trial,data):
    config = {
    "model_type": "single",
    "width_NN":  trial.suggest_categorical("width_NN", [8, 64, 256, 512]),
    "depth_NN":  trial.suggest_categorical("depth_NN", [1,4,8]),
    "modes":      trial.suggest_categorical("modes", [12,32,64]),
    "n_blocks":    trial.suggest_categorical("blocks", [1,4,8]),
    "block_act": trial.suggest_categorical(
                "block_act", [None, torch.nn.functional.tanh, torch.nn.functional.gelu, 
                              torch.nn.functional.sigmoid, torch.nn.functional.mish]),
    "lift_act": trial.suggest_categorical(
                "lift_act", [None, torch.nn.functional.tanh, torch.nn.functional.gelu, 
                             torch.nn.functional.sigmoid, torch.nn.functional.mish]),
    "width":     trial.suggest_categorical("width", [1,8,32,64]),
    "lr":             trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    "padding":  trial.suggest_categorical("padding", [0,9,18])
    }

    _, data_out = train_model(config, data, device)
    val_loss = data_out["val_loss_history"][-1]
    train_loss = data_out["loss_history"][-1]
    print(f"[Trial {trial.number}] Train loss: {train_loss:.3e}, Val loss: {val_loss:.3e}, Overfit gap: {val_loss - train_loss:.3e}", flush = True)

    # Optional overfitting penalty (e.g. L2 gap penalty)
    penalty_lambda = 1.0  # You can tune this
    objective_value = val_loss + penalty_lambda * max(val_loss - train_loss, 0)

    return objective_value


if __name__ == "__main__":
    data, device = prepare_data(model_type = 'dual', 
                                state_only = False)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_dual(trial, data), n_trials=500)

    print("Best trial:")
    print(study.best_trial)
