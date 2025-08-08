import torch
import torch.nn as nn
import torch.optim as optim
from FNO import FNO1d, dual_FNO
from postprocess_utils import load_model
import torch.nn.utils as utils

def model_setup(config : dict, 
                data: dict, 
                device: torch.device):

    if config['model_type'] == 'single':
        in_channels = data['train_x_norm'].shape[1]
        out_channels = data['train_y_norm'].shape[1]
        model = FNO1d(
                in_channels       = in_channels,
                out_channels      = out_channels,
                modes             = config['fno']['mode'],
                width             = config['fno']['width'],
                block_activation  = config['fno']['act'],
                n_blocks          = config['fno']['blocks'],
                padding           = config['fno']['padding'],
                coord_features    = config['fno']['coord_features'],
                adaptive          = config['fno']['adaptive'],
                lift_activation   = config['lift']['act'],
                lift_NN           = config['lift']['NN'],
                lift_NN_params    = config['lift']['NN_params'] if 'NN_params' in config['lift'] else None,
                decode_activation = config['decode']['act'],
                decode_NN         = config['decode']['NN'],
                decode_NN_params  = config['decode']['NN_params'] if 'NN_params' in config['decode'] else None,
                ).to(device)
    
    else:
        fno_heal = FNO1d(
                in_channels       = 1,
                out_channels      = 1,
                modes             = config['heal']['fno']['mode'],
                width             = config['heal']['fno']['width'],
                block_activation  = config['heal']['fno']['act'],
                n_blocks          = config['heal']['fno']['blocks'],
                padding           = config['heal']['fno']['padding'],
                coord_features    = config['heal']['fno']['coord_features'],
                adaptive          = config['heal']['fno']['adaptive'],
                lift_activation   = config['heal']['lift']['act'],
                lift_NN           = config['heal']['lift']['NN'],
                lift_NN_params    = config['heal']['lift']['NN_params'] if 'NN_params' in config['heal']['lift'] else {},
                decode_activation = config['heal']['decode']['act'],
                decode_NN         = config['heal']['decode']['NN'],
                decode_NN_params  = config['heal']['decode']['NN_params'] if 'NN_params' in config['heal']['decode'] else {},
                ).to(device)

        fno_state = FNO1d(
                in_channels       = 1,
                out_channels      = 1,
                modes             = config['state']['fno']['mode'],
                width             = config['state']['fno']['width'],
                block_activation  = config['state']['fno']['act'],
                n_blocks          = config['state']['fno']['blocks'],
                padding           = config['state']['fno']['padding'],
                coord_features    = config['state']['fno']['coord_features'],
                adaptive          = config['state']['fno']['adaptive'],
                lift_activation   = config['state']['lift']['act'],
                lift_NN           = config['state']['lift']['NN'],
                lift_NN_params    = config['state']['lift']['NN_params'] if 'NN_params' in config['state']['lift'] else {},
                decode_activation = config['state']['decode']['act'],
                decode_NN         = config['state']['decode']['NN'],
                decode_NN_params  = config['state']['decode']['NN_params'] if 'NN_params' in config['state']['decode'] else {},
                ).to(device)

        model = dual_FNO(fno_heal, fno_state).to(device)

    return model

def make_decay_fn(factor, interval):
    return lambda step: factor ** (step // interval)

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
            
            rand_int = [1,2,3,4][torch.multinomial(torch.Tensor([0.6,0.25,0.10,0.05]), 1).item()]
            batch_x_down = batch_x[:,:,::rand_int]
            
            std = torch.rand(1).item() * 0.015
            noise = torch.randn_like(batch_x_down) * std
            batch_x_noisy = batch_x_down + noise
            
            output = model(batch_x_noisy)
            loss = criterion(output, batch_y[:,:,::rand_int])
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_loss += loss.item()
            scheduler.step()

        if save_results or epoch == epochs -1:
            loss_history.append(epoch_loss / len(train_loader))
            
            model.eval()
            with torch.no_grad():
                val_output = model(val_x)
                val_loss = criterion(val_output, val_y).item()
                val_loss_history.append(val_loss)

    return loss_history, val_loss_history

def pretraining(model, 
                data, 
                device, 
                config, 
                criterion):
    
    if config['pretrain']['load'] == True:
        print("\t LOADING PRETRAINED")
        fno_state = load_model("state_model", device)
        model.FNO_state = fno_state

    else:
        print(f'\t PRETRAINING for {config['pretrain']['epochs']}')
        train_loader_state = data['train_loader_SO']
        test_x_state = data['test_x_norm_SO']
        test_y_state = data['test_y_norm_SO']

        optimizer_state = optim.Adam(model.FNO_state.parameters(), lr=config['pretrain']['lr'])
        scheduler_state = torch.optim.lr_scheduler.LambdaLR(optimizer_state, lr_lambda=make_decay_fn(0.9, 1000))

        loss_history_state, val_loss_history_state = train_loop(model.FNO_state, train_loader_state, test_x_state, test_y_state,
                                                                optimizer_state, scheduler_state, config['pretrain']['epochs'], 
                                                                criterion, config['pretrain']['save_results'])

        data['loss_history_state'] = loss_history_state
        data['val_loss_history_state'] = val_loss_history_state
        

def train_model(config, model, data, device="cuda"):
    print(f"Beginning Training for {config['train']['epochs']}")
    criterion = nn.MSELoss()
    lr = config['train']["lr"]

    if config['model_type'] == 'dual':

        if 'pretrain' in config:
            pretraining(model, data, device, config, criterion)

        lr_state_factor = config['train']["lr_state_factor"] if "lr_state_factor" in config['train'] else 1

        optimizer = optim.Adam([
            {"params": model.FNO_state.parameters(), "lr": lr * lr_state_factor, "weight_decay": 1e-4},
            {"params": model.FNO_heal.parameters(), "lr": lr, "weight_decay": 1e-4}
        ])

        if lr_state_factor < 1e-6:
            for param in model.FNO_state.parameters():
                param.requires_grad = False

    else: 
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=make_decay_fn(0.8, 1000))

    loss_history, val_loss_history = [], []
    train_loader, test_x, test_y = data['train_loader'], data['test_x_norm'], data['test_y_norm']

    loss_history, val_loss_history = train_loop(model, train_loader, test_x, test_y,
                                                optimizer, scheduler, config['train']['epochs'], 
                                                criterion, config['train']['save_results'])

    data['loss_history'] = loss_history
    data['val_loss_history'] = val_loss_history
    print("\t End of Training")