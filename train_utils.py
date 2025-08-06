import torch
import torch.nn as nn
import torch.optim as optim
from FNO import FNO1d, dual_FNO
from postprocess_utils import load_model

def model_setup(config, data, device):

    in_channels = data['train_x_norm'].shape[1]
    out_channels = data['train_y_norm'].shape[1]

    if config['train']['model_type'] == 'single':
        model = FNO1d(
                in_channels       = in_channels,
                out_channels      = out_channels,
                modes             = config['fno']['mode'],
                width             = config['fno']['width'],
                block_activation  = config['fno']['act'],
                n_blocks          = config['fno']['blocks'],
                padding           = config['fno']['padding'],
                coord_features    = config['fno']['coord_features'],
                lift_activation   = config['lift']['act'],
                lift_NN           = config['lift']['NN'],
                lift_NN_params    = config['lift']['NN_params'] if 'NN_params' in config['lift'] else None,
                decode_activation = config['decode']['act'],
                decode_NN         = config['decode']['NN'],
                decode_NN_params  = config['decode']['NN_params'] if 'NN_params' in config['decode'] else None,
                ).to(device)
    
    else:
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

    return model

def train_loop(model, train_loader, val_x, val_y, 
               optimizer, scheduler, epochs, 
               criterion = nn.MSELoss(), save_results = True):

        loss_history = []
        val_loss_history = []

        for epoch in range(epochs):
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

            if save_results or epoch == epochs -1:
                loss_history.append(epoch_loss / len(train_loader))
                
                model.eval()
                with torch.no_grad():
                    val_output = model(val_x)
                    val_loss = criterion(val_output, val_y).item()
                    val_loss_history.append(val_loss)

        return loss_history, val_loss_history

def pretraining(model, data, device, config, criterion):
    if config['pretrain']['load'] == True:
        print("LOADING PRETRAINED")
        fno_state = load_model("state_model", device)
        model.FNO_state = fno_state

    elif config['pretrain'] == 'load':
        print('PRETRAINING')
        train_loader_state = data['train_loader_state']
        test_x_state = data['test_x_norm_state']
        test_y_state = data['test_y_norm_state']

        optimizer_state = optim.Adam(model.FNO_state.parameters(), lr=config['pretrain']['lr'])
        scheduler_state = optim.lr_scheduler.ExponentialLR(optimizer_state, gamma=0.95)

        loss_history_state, val_loss_history_state = train_loop(model.FNO_state, train_loader_state, test_x_state, test_y_state,
                                                                optimizer_state, scheduler_state, config['pretrain']['epochs'], 
                                                                criterion, config['pretrain']['save_results'])

        data['loss_history_state'] = loss_history_state
        data['val_loss_history_state'] = val_loss_history_state
        

def train_model(config, model, data, device="cuda"):
    criterion = nn.MSELoss()
    lr = config['train']["lr"]

    if config['train']['model_type'] == 'dual':

        if 'pretrain' in config:
            pretraining(model, data, device, config, criterion)

        lr_state_factor = config['train']["lr_state_factor"] if "lr_state_factor" in config['train'] else 1

        optimizer = optim.Adam([
            {"params": model.FNO_state.parameters(), "lr": lr * lr_state_factor},
            {"params": model.FNO_heal.parameters(), "lr": lr}
        ])

        if config["lr_state_factor"] < 1e-6:
            for param in model.FNO_state.parameters():
                param.requires_grad = False

    else: 
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    loss_history, val_loss_history = [], []
    train_loader, test_x, test_y = data['train_loader'], data['test_x_norm'], data['test_y_norm']

    loss_history, val_loss_history = train_loop(model, train_loader, test_x, test_y,
                                                optimizer, scheduler, config['train']['epochs'], 
                                                criterion, config['train']['save_results'])

    data['loss_history'] = loss_history
    data['val_loss_history'] = val_loss_history