import torch
from FNO_torch import FNO1d, dual_FNO
from postprocess_utils import plot_results, plot_loss
from preprocess_utils import prepare_data

data, device = prepare_data('single', state_only = True)

checkpoint = torch.load('state_model.pth', map_location=device)
fno_state_params = checkpoint['model_params']
fno_state = FNO1d(**fno_state_params).to(device)
fno_state.load_state_dict(checkpoint['model_state_dict'])

plot_results(fno_state,data['train_x_norm'], data['train_y_norm'], 'state_train.png')
plot_results(fno_state,data['test_x_norm'], data['test_y_norm'], 'state_test.png')