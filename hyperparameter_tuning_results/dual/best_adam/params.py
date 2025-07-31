import torch

best_adam_params  = {
    "model_type": "dual",
    "width_NN_heal": 90,
    "width_NN_state": 135,
    "depth_NN_heal": 6,
    "depth_NN_state": 4,
    "mode_heal": 64,
    "mode_state": 64,
    "blocks_heal": 6,
    "blocks_state": 4,
    "lift_act_state": torch.tanh,
    "block_act_state": torch.nn.functional.gelu,
    "lift_act_heal": torch.tanh,
    "block_act_heal": torch.tanh,
    "width_heal": 17,
    "width_state": 13,
    "lr": 0.004169535105824954,
    "padding_heal": 3,
    "padding_state": 22,
    "optimizer_type": "adam"
}