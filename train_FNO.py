import torch
import numpy as np
from train_utils import train_model, model_setup
from preprocess_utils import prepare_data, combine_with_suffix
from postprocess_utils import plots, plot_results, plot_loss
from FNO import FNO1d
import torch.nn.functional as F 
import torch.nn as nn
import matplotlib.pyplot as plt

data_config = {"x_path": "data/friction_data/features_AgingLaw_v2.csv",
               "y_path": "data/friction_data/targets_AgingLaw_v2.csv",
               'train_samples': 600,
               'log_norm': True,
               'state_norm': False,
               'heal_norm': False}

lift_config = {"NN" : False,
               "act": nn.GELU()
               }

decode_config = {"NN": True,
                 "NN_params": {"width": 32,
                               "depth": 1},
                 "act": F.silu
                 }

fno_config = {"mode": 16,
              "blocks": 4,
              "act": F.gelu,
              "width": 32,
              "padding": 9,
              "coord_features": True
              }

train_config = {'lr': 1e-3,
                'save_results': True,
                'epochs': 100,
                'model_type': 'single'}

config = {"train": train_config,
          "lift": lift_config,
          "fno": fno_config,
          "decode": decode_config}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = prepare_data(data_config, device)
    model = model_setup(config, data, device)
    train_model(config, model, data, device)
    plots(model, data, device)    



