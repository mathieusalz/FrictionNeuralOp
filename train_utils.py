import torch
import torch.nn as nn
import torch.optim as optim
from FNO_torch import FNO1d, dual_FNO
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

def prepare_data(model_type):
    feature = np.genfromtxt('friction_data/features_AgingLaw_v2.csv', delimiter=',')[:, np.newaxis, :]
    target = np.genfromtxt('friction_data/targets_AgingLaw_v2.csv', delimiter=',')[:, np.newaxis, :]

    train_x = feature[:750, :, :]
    train_y = target[:750, :, :]
    test_x = feature[750:, :, :]
    test_y = target[750:, :, :]

    if model_type == 'dual':
        train_x_norm_state = torch.Tensor(np.sqrt(np.log(1e8*train_x)))
        x_max = train_x_norm_state.max()
        train_x_norm_state /= x_max
        train_x_norm_heal = torch.Tensor(np.exp(-1e7*train_x))
        train_x_norm = torch.cat([train_x_norm_state, train_x_norm_heal], dim=1)
    
        test_x_norm_state = torch.Tensor(np.sqrt(np.log(1e8*test_x))) / x_max
        test_x_norm_heal = torch.Tensor(np.exp(-1e7*test_x))
        test_x_norm = torch.cat([test_x_norm_state, test_x_norm_heal], dim=1)
    else:
        train_x_norm = torch.Tensor(np.log(np.log(1/train_x)))
        test_x_norm = torch.Tensor(np.log(np.log(1 / test_x)))

    train_y_norm = torch.Tensor(train_y / train_y.max())
    test_y_norm = torch.Tensor(test_y / train_y.max())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TensorDataset(train_x_norm.to(device), train_y_norm.to(device))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    return train_loader, train_x_norm.to(device), train_y_norm.to(device), test_x_norm.to(device), test_y_norm.to(device), device

def train_model(config, train_loader, test_x, test_y, device="cuda", save_results = False, total_epochs = 80):
    
    loss_history, val_loss_history = [], []

    if config['model_type'] == 'dual':

        fno_heal = FNO1d(
            in_channels=1,
            out_channels=1,
            modes=config["mode_heal"],
            width=config["width_heal"],
            block_activation=config["block_act_heal"],
            lifting_activation=config["lift_act_heal"],
            n_blocks=config["blocks_heal"],
            padding=config["padding_heal"],
            NN=True,
            NN_params={"width": config["width_NN_heal"], "depth": config["depth_NN_heal"]}
        ).to(device)

        fno_state = FNO1d(
            in_channels=1,
            out_channels=1,
            modes=config["mode_state"],
            width=config["width_heal"],
            block_activation=config["block_act_state"],
            lifting_activation=config["lift_act_state"],
            n_blocks=config["blocks_state"],
            padding=config["padding_state"],
            NN=True,
            NN_params={"width": config["width_NN_state"], "depth": config["depth_NN_state"]}
        ).to(device)

        model = dual_FNO(fno_heal, fno_state).to(device)

    else: 

        model = FNO1d(
            in_channels=1,
            out_channels=1,
            modes= config['modes'],
            width= config['width'],
            block_activation=config['block_act'],
            lifting_activation=config['lift_act'],
            n_blocks=config['n_blocks'],
            padding=config['padding'],
            NN=True,
            NN_params={"width": config['width_NN'], "depth": config['depth_NN']}
            ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if scheduler:
            scheduler.step()


        if save_results or epoch == total_epochs -1:
            loss_history.append(epoch_loss / len(train_loader))
            
            model.eval()
            with torch.no_grad():
                val_output = model(test_x)
                val_loss = criterion(val_output, test_y).item()
                val_loss_history.append(val_loss)

    return model, loss_history, val_loss_history

def plots(model, data, device):

    loss_history = data['loss_history']
    val_loss_history = data['val_loss_history']
    train_x_norm = data['train_x_norm']
    train_y_norm = data['train_y_norm']
    test_x_norm = data['test_x_norm']
    test_y_norm = data['test_y_norm']

    # PLOT 1 - Loss plot
    plt.figure(figsize=(8,6))
    plt.plot(loss_history, label="train loss")
    plt.plot(val_loss_history, label="val loss")
    plt.legend()
    plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.savefig("loss_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2 - Inference on Training Set
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    axes = axes.flatten()
    for i in range(100):
        pred = model(train_x_norm[i:i+1])[0].cpu().detach().numpy()
        ax1 = axes[i]

        ax1.plot(train_y_norm[i, 0, :].cpu().detach().numpy(), color='tab:orange', label='Ground Truth' if i == 0 else "")
        ax1.plot(pred[0], color='tab:green', label='Prediction' if i == 0 else "")
        ax1.set_xticks([])
        ax1.set_yticks([])

    fig.legend(loc='upper center', ncol=2, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for legend on top
    fig.savefig("train_inference.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot 3 - Inference on Testing Set
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    axes = axes.flatten()
    for i in range(100):
        pred = model(test_x_norm[i:i+1])[0].cpu().detach().numpy()
        ax1 = axes[i]

        ax1.plot(test_y_norm[i, 0, :].cpu().detach().numpy(), color='tab:orange', label='Ground Truth' if i == 0 else "")
        ax1.plot(pred[0], color='tab:green', label='Prediction' if i == 0 else "")
        ax1.set_xticks([])
        ax1.set_yticks([])

    fig.legend(loc='upper center', ncol=2, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("test_inference.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
