import numpy as np
import matplotlib.pyplot as plt
from FNO import FNO1d, dual_FNO
import torch

def plot_results(model, x, y, name):
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    axes = axes.flatten()
    for i in range(100):
        pred = model(x[i:i+1])[0].cpu().detach().numpy()
        ax1 = axes[i]

        ax1.plot(y[i, 0, :].cpu().detach().numpy(), color='tab:orange', label='Ground Truth' if i == 0 else "")
        ax1.plot(pred[0], color='tab:green', label='Prediction' if i == 0 else "")
        ax1.set_xticks([])
        ax1.set_yticks([])

    fig.legend(loc='upper center', ncol=2, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for legend on top
    fig.savefig(name, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_loss(loss, val, name):
    plt.figure(figsize=(8,6))
    plt.plot(loss, label="train loss")
    plt.plot(val, label="val loss")
    plt.legend()
    plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.close()


def plots(model, data, device):
    
    train_x_norm = data['train_x_norm']
    train_y_norm = data['train_y_norm']
    test_x_norm = data['test_x_norm']
    test_y_norm = data['test_y_norm']

    # PLOT 1 - Loss plot
    if 'loss_history' in data:
        loss_history = data['loss_history']
        val_loss_history = data['val_loss_history']
        plot_loss(loss_history, val_loss_history, "loss_plot.png")
    
    # Plot 2 - Inference on Training Set
    plot_results(model, train_x_norm, train_y_norm, "train_inference.png")
    
    # Plot 3 - Inference on Testing Set
    plot_results(model, test_x_norm, test_y_norm, "test_inference.png")
     
    if isinstance(model, dual_FNO):
        if 'loss_history_state' in data:
            plot_loss(data['loss_history_state'], data['val_loss_history_state'], "loss_plot_pretrain.png")
        plot_results(model.FNO_state,data['train_x_norm_SO'], data['train_y_norm_SO'], 'state_train.png')
        plot_results(model.FNO_state,data['test_x_norm_SO'], data['test_y_norm_SO'], 'state_test.png')
        
        fig, axes = plt.subplots(6, 6, figsize=(20, 20))
        axes = axes.flatten()
        for i in range(36):
            pred_state = model.FNO_state(train_x_norm[i:i+1,0:1,:])[0].cpu().detach().numpy()
            pred_heal = model.FNO_heal(train_x_norm[i:i+1,1:2,:])[0].cpu().detach().numpy()
            ax1 = axes[i]

            ax1.plot(pred_state[0], color='tab:green', label='state' if i == 0 else "")
            ax1.plot(pred_heal[0], color='tab:blue', label='heal' if i == 0 else "")
            ax1.set_xticks([])

            ax2 = ax1.twinx()
            ax2.plot(data["train_x"][i,0,:], color= 'tab:red', label='input'if i==0 else "", linestyle = 'dashed')
            ax2.set_yscale('log')
            ax2.tick_params(axis='y', colors='tab:red')  # Larger y-ticks for secondary axis
            ax2.spines['right'].set_color('tab:red')
            ax2.set_xticks([])
            #ax1.set_yticks([])

        fig.legend(loc='upper center', ncol=3, fontsize=25)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for legend on top
        plt.savefig("train_contributions.png", dpi=300, bbox_inches='tight')
        plt.close()

def save_model(model, name = 'model'):

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_params': {
            'in_channels': model.in_channels,
            'out_channels': model.out_channels,
            'modes': model.modes,
            'width': model.width,
            'block_activation': model.block_activation,
            'lifting_activation': model.lifting_activation,
            'n_blocks': model.n_blocks,
            'padding': model.padding,
            'NN': model.NN,
            'NN_params': model.NN_params if model.NN else None,
            'bias': model.lifting.bias is not None if not model.NN else False,
        }
    }, f'{name}.pth')

def load_model(model_name, device):
    checkpoint = torch.load(f'{model_name}.pth', map_location=device)
    model_params = checkpoint['model_params']
    model = FNO1d(**model_params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model