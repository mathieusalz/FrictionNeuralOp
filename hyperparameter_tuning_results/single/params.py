import torch

params = {
    "model_type": 'single',
    "width_NN": 14,
    "depth_NN": 5,
    "modes": 60,
    "n_blocks": 6,
    "block_act": torch.nn.functional.gelu,
    "lift_act": torch.tanh,
    "width": 33,
    "lr": 0.0034354874600480943,
    "padding": 14
}