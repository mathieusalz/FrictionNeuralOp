import torch

best_lbfgs_params = {
    "model_type": 'dual',
    "width_NN_heal": 406,
    "width_NN_state": 283,
    "depth_NN_heal": 2,
    "depth_NN_state": 3,
    "mode_heal": 60,
    "mode_state": 56,
    "blocks_heal": 8,
    "blocks_state": 3,
    "lift_act_state": torch.nn.functional.gelu,
    "block_act_state": torch.nn.functional.gelu,
    "lift_act_heal": torch.nn.functional.gelu,
    "block_act_heal": None,
    "width_heal": 46,
    "width_state": 109,
    "lr": 0.0017813397998222417,
    "padding_heal": 1,
    "padding_state": 17,
    "optimizer_type": "lbfgs"
}