import torch
import numpy as np
from train_utils import train_model, model_setup
from preprocess_utils import prepare_data, combine_with_suffix
from postprocess_utils import plots, plot_results, plot_loss
from FNO_torch import FNO1d
import matplotlib.pyplot as plt

training_params = {"model_type": "dual",
                   "pretrain": False,
                   "lr": 2e-3,
                   "lr_state_factor": 1}

best_heal_params = {'width_NN': 64,
                     'depth_NN': 8,
                     'mode': 12,
                     'blocks': 6,
                     'lift_act': torch.nn.functional.tanh,
                     'block_act': torch.nn.functional.mish,
                     'width': 32,
                     'padding': 18}

best_state_params = {'width_NN': 256,
                     'depth_NN': 4,
                     'mode': 64,
                     'blocks': 2,
                     'lift_act': torch.nn.functional.mish,
                     'block_act': None,
                     'width': 64,
                     'padding': 18}

best_single_model = {"model_type": "single",
                     "mode": 64,
                     "blocks": 8,
                     "lift_act": torch.nn.functional.gelu,
                     "block_act": torch.nn.functional.tanh,
                     "width": 8,
                     "lr": 2e-3,
                     "padding": 9}

#config = {**training_params, **combine_with_suffix(best_heal_params, best_state_params)}

if __name__ == "__main__":
    data, device = prepare_data("dual")
    model = model_setup(best_single_model, device, NN= False)

    train_model(best_single_model, model, data, device, save_results = True, total_epochs= 100, NN = False)
    
    plots(model, data, device)    



