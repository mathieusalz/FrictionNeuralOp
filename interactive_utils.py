import numpy as np
import matplotlib.pyplot as plt
from FNO import FNO1d
import torch
from train_utils import train_loop, get_activation
import torch.optim as optim
import ipywidgets as widgets
from IPython.display import display, clear_output
from save_utils import save_model
from plot_utils import plot_loss, plot_results

# def interactive_training_ui(train_loader, train_x, train_y, test_x, test_y, out_of_dist_x, out_of_dist_y, epochs):
#     """
#     Creates an interactive training UI for a given model and dataset.
#     """
#     # --- Outputs ---
#     loss_out = widgets.Output()
#     train_out = widgets.Output()
#     test_out = widgets.Output()
#     ood_out = widgets.Output()
    
#     tab = widgets.Tab(children=[loss_out, train_out, test_out, ood_out])
#     tab.set_title(0, "Loss Histories")
#     tab.set_title(1, "Training Inference")
#     tab.set_title(2, "Testing Inference")
#     tab.set_title(3, "Out of Distribution")
    
#     # --- Shared state ---
#     state = {
#         "model": None,
#         "loss_history": None,
#         "val_loss_history": None,
#         "training": False
#     }
   
#     # --- Tab rendering ---
#     def render_tab(change=None):
#         if state["training"] or state["model"] is None:
#             return
#         idx = tab.selected_index if change is None else change["new"]
#         out_widget_map = [loss_out, train_out, test_out, ood_out]
#         data_map = [
#             (train_x, train_y),
#             (train_x, train_y),
#             (test_x, test_y),
#             (out_of_dist_x, out_of_dist_y)
#         ]
#         titles = ["Loss Histories", "Training Inference", "Testing Inference", "Out of Distribution"]
        
#         with out_widget_map[idx]:
#             clear_output(wait=True)
#             if idx == 0:
#                 fig = plot_loss(state["loss_history"], state["val_loss_history"], titles[idx])
#             else:
#                 x, y = data_map[idx]
#                 fig = plot_results(state["model"], x, y, titles[idx])
#             display(fig)
#             plt.close(fig)
    
#     def render_all_tabs():
#         if state["model"] is None:
#             return
#         for i in range(4):
#             render_tab({"new": i})
    
#     tab.observe(render_tab, names="selected_index")
    
#     # --- Training function ---
#     def train_model(change=None):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         state["training"] = True
        
#         for out_widget in [loss_out, train_out, test_out, ood_out]:
#             with out_widget:
#                 clear_output(wait=True)
#                 print("Training in progress...")
        
#         torch.manual_seed(0)
#         model = FNO1d(
#             in_channels=1,
#             out_channels=1,
#             modes=modes_slider.value,
#             width=width_slider.value,
#             n_blocks=n_blocks_slider.value,
#             block_activation=get_activation(block_drop.value),
#             lift_activation=get_activation(lift_drop.value),
#             decode_activation=get_activation(decode_drop.value)
#         ).to(device)
        
#         optimizer = optim.Adam(model.parameters(), lr=1e-2)
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
#         loss_history, val_loss_history = train_loop(
#             model=model,
#             train_loader=train_loader,
#             val_x=test_x,
#             val_y=test_y,
#             epochs=epochs,
#             optimizer=optimizer,
#             scheduler=scheduler,
#             verbose=False,
#             sample_freq=4
#         )
        
#         state.update({
#             "model": model,
#             "loss_history": loss_history,
#             "val_loss_history": val_loss_history,
#             "training": False
#         })
        
#         render_all_tabs()
    
#     # --- Controls ---
#     modes_slider = widgets.SelectionSlider(
#         options=[1, 2, 4, 8, 16, 32, 64, 128, 256],
#         value=32, description="Modes:", continuous_update=False
#     )
#     width_slider = widgets.SelectionSlider(
#         options=[1, 2, 4, 8, 16, 32, 64],
#         value=8, description="Width:", continuous_update=False
#     )
#     n_blocks_slider = widgets.SelectionSlider(
#         options=[1, 2, 4, 8],
#         value=1, description="Blocks:", continuous_update=False
#     )
#     sliders_row = widgets.HBox([modes_slider, width_slider, n_blocks_slider])

#     # --- Activation dropdowns ---
#     act_options = ['None', 'gelu', 'silu', 'relu', ]
#     block_drop = widgets.Dropdown(options=act_options, value='None', description='Block Act:')
#     lift_drop = widgets.Dropdown(options=act_options, value='None', description='Lift Act:')
#     decode_drop = widgets.Dropdown(options=act_options, value='None', description='Decode Act:')
    
#     dropdowns_row = widgets.HBox([block_drop, lift_drop, decode_drop])

#     # --- Save button ---
#     save_button = widgets.Button(description="Save Model", button_style='success')

#     def save_current_model(b):
#         if state["model"] is None:
#             print("No model to save!")
#             return
#         # Construct name from hyperparameters
#         name_parts = [
#             f"mode{modes_slider.value}",
#             f"width{width_slider.value}",
#             f"blocks{n_blocks_slider.value}",
#             f"block{block_drop.value}",
#             f"lift{lift_drop.value}",
#             f"decode{decode_drop.value}"
#         ]
#         model_name = "_".join(name_parts)
#         save_model(state["model"], model_name)
#         print(f"Model saved as '{model_name}'")

#     save_button.on_click(save_current_model)

#     # Add the button to the UI
#     ui = widgets.VBox([sliders_row, dropdowns_row, save_button])
    
#     # --- Initial training ---
#     train_model()
    
#     # --- Attach observers ---
#     for ctrl in [modes_slider, width_slider, n_blocks_slider, block_drop, lift_drop, decode_drop]:
#         ctrl.observe(train_model, names='value')
    
#     display(ui, tab)

def interactive_training_ui(train_loader, train_x, train_y, test_x, test_y, out_of_dist_x, out_of_dist_y, epochs):
    """
    Creates an interactive training UI for a given model and dataset.
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
            in_channels=1,
            out_channels=1,
            modes=modes_slider.value,
            width=width_slider.value,
            n_blocks=n_blocks_slider.value,
            block_activation=get_activation(block_drop.value),
            lift_activation=get_activation(lift_drop.value),
            decode_activation=get_activation(decode_drop.value)
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        loss_history, val_loss_history = train_loop(
            model=model,
            train_loader=train_loader,
            val_x=test_x,
            val_y=test_y,
            epochs=epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            verbose=True,
            sample_freq=4
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
        options=[1, 2, 4, 8],
        value=1, description="Blocks:", continuous_update=False
    )
    sliders_row = widgets.HBox([modes_slider, width_slider, n_blocks_slider])

    # --- Activation dropdowns ---
    act_options = ['None', 'gelu', 'silu', 'relu']
    block_drop = widgets.Dropdown(options=act_options, value='None', description='Block Act:')
    lift_drop = widgets.Dropdown(options=act_options, value='None', description='Lift Act:')
    decode_drop = widgets.Dropdown(options=act_options, value='None', description='Decode Act:')
    
    dropdowns_row = widgets.HBox([block_drop, lift_drop, decode_drop])

    # --- Buttons ---
    train_button = widgets.Button(description="Train Model", button_style='info')
    save_button = widgets.Button(description="Save Model", button_style='success')

    def train_model_button(b):
        train_model()

    train_button.on_click(train_model_button)

    def save_current_model(b):
        if state["model"] is None:
            print("No model to save!")
            return
        name_parts = [
            f"mode{modes_slider.value}",
            f"width{width_slider.value}",
            f"blocks{n_blocks_slider.value}",
            f"block{block_drop.value}",
            f"lift{lift_drop.value}",
            f"decode{decode_drop.value}"
        ]
        model_name = "_".join(name_parts)
        save_model(state["model"], model_name)
        print(f"Model saved as '{model_name}'")

    save_button.on_click(save_current_model)

    buttons_row = widgets.HBox([train_button, save_button])

    # --- Final UI ---
    ui = widgets.VBox([sliders_row, dropdowns_row, buttons_row])
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
