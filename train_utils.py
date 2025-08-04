import torch
import torch.nn as nn
import torch.optim as optim
from FNO_torch import FNO1d, dual_FNO

def single_model_setup(config, device):

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

    return model    

def dual_model_setup(config, device):
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

def train_model(config, data, device="cuda", save_results = False, total_epochs = 80):
    criterion = nn.MSELoss()

    if config['model_type'] == 'dual':
        model = dual_model_setup(config, device)
    else: 
        model = single_model_setup(config, device)
        
    if 'pretrain' in config:
        if config['pretrain'] == True:
            print('PRETRAINING')
            train_loader_state = data['train_loader_state']
            test_x_state = data['test_x_norm_state']
            test_y_state = data['test_y_norm_state']

            optimizer_state = optim.Adam(model.FNO_state.parameters(), lr=config['lr'])
            scheduler_state = optim.lr_scheduler.ExponentialLR(optimizer_state, gamma=0.95)

            loss_history_state, val_loss_history_state = train_loop(model.FNO_state, train_loader_state, test_x_state, test_y_state,
                                                                    optimizer_state, scheduler_state, config['pretrain_epochs'], 
                                                                    criterion, save_results)

            data['loss_history_state'] = loss_history_state
            data['val_loss_history_state'] = val_loss_history_state

        elif config['pretrain'] == 'load':
            print("LOADING PRETRAINED")
            checkpoint = torch.load('state_model.pth', map_location=device)
            fno_state_params = checkpoint['model_params']
            fno_state = FNO1d(**fno_state_params).to(device)
            fno_state.load_state_dict(checkpoint['model_state_dict'])
            model.FNO_state = fno_state

    loss_history = []
    val_loss_history = []

    if config['model_type'] == 'dual':

        optimizer = optim.Adam([
            {"params": model.FNO_state.parameters(), "lr": config["lr"] * config["lr_state_factor"]},
            {"params": model.FNO_heal.parameters(), "lr": config["lr"]}
        ])

        if config["lr_state_factor"] < 1e-6:
            print("Not Training Pretrained")
            for param in model.FNO_state.parameters():
                param.requires_grad = False
    else:
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    train_loader = data['train_loader']
    test_x = data['test_x_norm']
    test_y = data['test_y_norm']

    loss_history, val_loss_history = train_loop(model, train_loader, test_x, test_y,
                                                optimizer, scheduler, total_epochs, 
                                                criterion, save_results)

    data['loss_history'] = loss_history
    data['val_loss_history'] = val_loss_history
    return model, data
