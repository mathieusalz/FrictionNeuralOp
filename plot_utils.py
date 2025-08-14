import numpy as np
import matplotlib.pyplot as plt
from FNO import FNO1d, dual_FNO
import torch
from train_utils import train_loop
import torch.optim as optim
import ipywidgets as widgets
from IPython.display import display, clear_output


def plot_data(x, y, name):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(16):
        ax1 = axes[i]
        ax1.plot(x[i], label='Input' if i == 0 else "")
        ax1.plot(y[i], label='Output' if i == 0 else "")
        ax1.set_xticks([])
        ax1.tick_params(axis='y', labelsize=14)  # Bigger y-ticks

    fig.suptitle(name, fontsize=22, y=0.98)  # Title closer to top

    # Legend below the title, slightly lower than before
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space for title+legend
    plt.show()


def plot_results(model: torch.nn.Module, 
                 x: torch.Tensor, 
                 y: torch.Tensor, 
                 name: str,
                 save: bool = False) -> None:
    """
    Plot model predictions versus ground truth for 100 samples and save the figure.

    Args:
        model (torch.nn.Module): Trained model used for inference.
        x (torch.Tensor): Input tensor to generate predictions.
        y (torch.Tensor): Ground truth tensor for comparison.
        name (str): Filename to save the generated plot.
    """
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    for i in range(16):
        pred = model(x[i:i+1])[0].cpu().detach().numpy()
        ax1 = axes[i]

        ax1.plot(y[i, 0, :].cpu().detach().numpy(), color='tab:orange', label='Ground Truth' if i == 0 else "")
        ax1.plot(pred[0], color='tab:green', label='Prediction' if i == 0 else "")
        ax1.set_xticks([])
        ax1.set_yticks([])

    fig.suptitle(name, fontsize=22, y=0.98)  # Title closer to top

    # Legend below the title, slightly lower than before
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for legend on top
    if save:
        fig.savefig(name, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig

def plot_loss(loss: list, val: list, name: str, save:bool = False) -> None:
    """
    Plot training and validation loss curves on a logarithmic scale and save the figure.

    Args:
        loss (list): List of training loss values per epoch.
        val (list): List of validation loss values per epoch.
        name (str): Filename to save the loss plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(loss, label="train loss")
    ax.plot(val, label="val loss")
    ax.legend(fontsize=14)
    ax.set_yscale("log")
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.suptitle(name, fontsize=18)
    plt.tight_layout()
    return fig

def plots(model: torch.nn.Module, data: dict, device: torch.device) -> None:
    """
    Generate and save multiple plots including loss curves and inference results for train/test sets.

    If the model is a dual_FNO, additionally plots pretraining losses and separate predictions for state and heal components.

    Args:
        model (torch.nn.Module): The trained model (single or dual).
        data (dict): Dataset dictionary containing normalized inputs, targets, and loss histories.
        device (torch.device): Device where tensors reside.
    """    
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

def interactive_training_ui(train_loader, train_x, train_y, test_x, test_y, out_of_dist_x, out_of_dist_y):
    """
    Creates an interactive training UI for a given model and dataset.
    
    Args:
        train_x, train_y: Training data
        test_x, test_y: Validation/testing data
        out_of_dist_x, out_of_dist_y: Out-of-distribution data
        train_loop: Function to train the model
        FNO1d: Model class to instantiate
    """
    # --- Outputs ---
    loss_out = widgets.Output()
    train_out = widgets.Output()
    test_out = widgets.Output()
    ood_out = widgets.Output()
    
    tab = widgets.Tab(children=[loss_out, train_out, test_out, ood_out])
    tab.set_title(0, "Loss Histories")
    tab.set_title(1, "Training Inference")
    tab.set_title(2, "Testing Inference")
    tab.set_title(3, "Out of Distribution")
    
    # --- Shared state ---
    state = {
        "model": None,
        "loss_history": None,
        "val_loss_history": None,
        "training": False
    }
    
    # --- Tab rendering ---
    def render_tab(change=None):
        if state["training"] or state["model"] is None:
            return
        idx = tab.selected_index if change is None else change["new"]
        out_widget_map = [loss_out, train_out, test_out, ood_out]
        data_map = [
            (train_x, train_y),
            (train_x, train_y),
            (test_x, test_y),
            (out_of_dist_x, out_of_dist_y)
        ]
        titles = ["Loss Histories", "Training Inference", "Testing Inference", "Out of Distribution"]
        
        with out_widget_map[idx]:
            clear_output(wait=True)
            if idx == 0:
                fig = plot_loss(state["loss_history"], state["val_loss_history"], titles[idx])
            else:
                x, y = data_map[idx]
                fig = plot_results(state["model"], x, y, titles[idx])
            display(fig)
            plt.close(fig)
    
    def render_all_tabs():
        """Render all tabs after training completes"""
        if state["model"] is None:
            return
        for i in range(4):
            render_tab({"new": i})

    
    tab.observe(render_tab, names="selected_index")
    
    # --- Training function ---
    def train_model(change=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state["training"] = True
        
        for out_widget in [loss_out, train_out, test_out, ood_out]:
            with out_widget:
                clear_output(wait=True)
                print("Training in progress...")
        
        torch.manual_seed(0)
        model = FNO1d(
            in_channels=1, out_channels=1,
            modes=modes_slider.value,
            width=width_slider.value,
            n_blocks=n_blocks_slider.value
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        loss_history, val_loss_history = train_loop(
            model=model,
            train_loader=train_loader,
            val_x=test_x,
            val_y=test_y,
            epochs=20,
            optimizer=optimizer,
            scheduler=scheduler,
            verbose=False,
            sample_freq = 4
        )
        
        state.update({
            "model": model,
            "loss_history": loss_history,
            "val_loss_history": val_loss_history,
            "training": False
        })
        
        render_all_tabs()
    
    # --- Controls ---
    modes_slider = widgets.SelectionSlider(
        options=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        value=32, description="Modes:", continuous_update=False
    )
    width_slider = widgets.SelectionSlider(
        options=[1, 2, 4, 8, 16, 32, 64],
        value=8, description="Width:", continuous_update=False
    )
    n_blocks_slider = widgets.SelectionSlider(
        options=[1, 2, 4, 8], value=1, description="Blocks:", continuous_update=False
    )
    
    ui = widgets.HBox([modes_slider, width_slider, n_blocks_slider])
    
    # --- Initial training ---
    train_model()
    
    # --- Attach slider observers ---
    for slider in [modes_slider, width_slider, n_blocks_slider]:
        slider.observe(train_model, names='value')
    
    display(ui, tab)

def interactive_sampling_plot(model, train_x, train_y, test_x, test_y, out_of_dist_x, out_of_dist_y, training_resolution, modes):
    """
    Interactive plot of FNO predictions for multiple samples with variable upsampling and downsampling.
    Each tab contains four rows of chosen samples. The first subplot uses an interactive upsample slider, 
    the second uses the training resolution, the third uses an interactive downsample slider, and the fourth shows an overlay.
    """

    # Hardcoded samples
    out_of_dist_to_plot = [
        (out_of_dist_y[0,0,:], out_of_dist_x[0,0,:]),
        (out_of_dist_y[4,0,:], out_of_dist_x[4,0,:]),
        (out_of_dist_y[8,0,:], out_of_dist_x[8,0,:]),
        (out_of_dist_y[12,0,:], out_of_dist_x[12,0,:])
    ]

    train_to_plot = [
        (train_y[0,0,:], train_x[0,0,:]),
        (train_y[1,0,:], train_x[1,0,:]),
        (train_y[2,0,:], train_x[2,0,:]),
        (train_y[3,0,:], train_x[3,0,:])
    ]

    test_to_plot = [
        (test_y[0,0,:], test_x[0,0,:]),
        (test_y[1,0,:], test_x[1,0,:]),
        (test_y[2,0,:], test_x[2,0,:]),
        (test_y[3,0,:], test_x[3,0,:])
    ]

    def plot_samples(tab_name, upsample_factor, downsample_factor):
        if tab_name == "Train":
            samples = train_to_plot
        elif tab_name == "Test":
            samples = test_to_plot
        elif tab_name == "OOD":
            samples = out_of_dist_to_plot
        else:
            return

        fig, axes = plt.subplots(len(samples), 4, figsize=(16, 3*len(samples)))

        if len(samples) == 1:
            axes = axes[np.newaxis, :]  # Ensure 2D array for consistency

        for row_idx, (y_sample, x_sample) in enumerate(samples):
            # Subplot 1: Upsampled at slider value
            real_upsample_factor = training_resolution - upsample_factor if upsample_factor - training_resolution != 0 else 1
            axes[row_idx, 0].scatter(np.linspace(0,1,len(x_sample[::real_upsample_factor])), x_sample[::real_upsample_factor], label="Input x")
            axes[row_idx, 0].plot(np.linspace(0,1,len(y_sample[::real_upsample_factor])),
                                   model(x_sample[None,None,::real_upsample_factor])[0][0].cpu().detach().numpy(),
                                   color='red', label="Prediction")
            axes[row_idx, 0].set_title(f"Upsampled at x {upsample_factor}" if row_idx==0 else "")
            axes[row_idx, 0].legend()

            # Subplot 2: Sampled at training_resolution
            axes[row_idx, 1].scatter(np.linspace(0,1,len(x_sample[::training_resolution])), x_sample[::training_resolution], label="Input x")
            axes[row_idx, 1].plot(np.linspace(0,1,len(y_sample[::training_resolution])),
                                   model(x_sample[None,None,::training_resolution])[0][0].cpu().detach().numpy(),
                                   color='red', label="Prediction")
            axes[row_idx, 1].set_title(f"Sampled at Training Resolution" if row_idx==0 else "")
            axes[row_idx, 1].legend()

            # Subplot 3: Downsampled at slider value
            real_downsample_factor = training_resolution * downsample_factor
            axes[row_idx, 2].scatter(np.linspace(0,1,len(x_sample[::real_downsample_factor])), x_sample[::real_downsample_factor], label="Input x")
            axes[row_idx, 2].plot(np.linspace(0,1,len(y_sample[::real_downsample_factor])),
                                   model(x_sample[None,None,::real_downsample_factor])[0][0].cpu().detach().numpy(),
                                   color='red', label="Prediction")
            axes[row_idx, 2].set_title(f"Downsampled at x {downsample_factor}" if row_idx==0 else "")
            axes[row_idx, 2].legend()

            # Subplot 4: Overlay of all resolutions + target y
            axes[row_idx, 3].plot(np.linspace(0,1,len(y_sample)), y_sample, color='black', linestyle='--', label="Target y")
            axes[row_idx, 3].plot(np.linspace(0,1,len(y_sample[::real_upsample_factor])), model(x_sample[None,None,::real_upsample_factor])[0][0].cpu().detach().numpy(), label=f"Upsampled {upsample_factor}")
            axes[row_idx, 3].plot(np.linspace(0,1,len(y_sample[::training_resolution])), model(x_sample[None,None,::training_resolution])[0][0].cpu().detach().numpy(), label=f"Sampled {training_resolution}")
            axes[row_idx, 3].plot(np.linspace(0,1,len(y_sample[::real_downsample_factor])), model(x_sample[None,None,::real_downsample_factor])[0][0].cpu().detach().numpy(), label=f"Downsampled {downsample_factor}")
            axes[row_idx, 3].set_title("Overlay" if row_idx==0 else "")
            if row_idx==0:
                axes[row_idx, 3].legend()

        plt.tight_layout()
        plt.show()

    tab_selector = widgets.Dropdown(
        options=["Train", "Test", "OOD"],
        value="OOD",
        description="Tab:"
    )
    upsample_slider = widgets.IntSlider(value=2, min=2, max=training_resolution, step=1, description="Upsample")
    downsample_slider = widgets.IntSlider(value= 4, min=2, max=int(2000/(2 *training_resolution * (modes+1))), step=1, description="Downsample")

    widgets.interact(plot_samples, tab_name=tab_selector, upsample_factor=upsample_slider, downsample_factor=downsample_slider)
