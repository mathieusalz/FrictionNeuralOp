import torch
import numpy as np
from train_utils import train_model, model_setup
from preprocess_utils import prepare_data, combine_with_suffix
from postprocess_utils import plots, plot_results, plot_loss
from FNO import FNO1d
import torch.nn as nn
from copy import deepcopy as dc


'''
SiLU
GELU
Tanh
Mish
LeakyReLU
ELU
'''

data_config = {"x_path": "data/friction_data/features_AgingLaw_v2.csv",
               "y_path": "data/friction_data/targets_AgingLaw_v2.csv",
               'train_samples': 700,
               'log_norm': True,
               'state_norm': True,
               'heal_norm': True}

lift_config = {"NN" : False,
               "act": nn.GELU()
               }

# lift_config = {"NN": True,
#                  "NN_params": {"width": 32,
#                                "depth": 1},
#                  "act": nn.SiLU()
#                  }

decode_config = {"NN": True,
                 "NN_params": {"width": 32,
                               "depth": 1},
                 "act": nn.SiLU()
                 }

fno_config = {"mode": 16,
              "blocks": 4,
              "act": nn.GELU(),
              "width": 32,
              "padding": 9,
              "coord_features": True
              }

train_config = {'lr': 1e-3,
                'save_results': True,
                'epochs': 400,
                'model_type': 'single'}

# config = {"train": train_config,
#           "heal": {"fno" : fno_config,
#                    "lift" : lift_config,
#                    "decode": decode_config},
#           "state": {"fno" : dc(fno_config),
#                    "lift" : dc(lift_config),
#                    "decode": dc(decode_config)}
#           }

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



