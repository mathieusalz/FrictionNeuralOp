import torch
import numpy as np
from train_utils import train_model, prepare_data, plots


params = {}

if __name__ == "__main__":
    train_loader, train_x_norm, train_y_norm, test_x_norm, test_y_norm, device = prepare_data()

    model, loss_history, val_loss_history = train_model(params, train_loader, test_x_norm, test_y_norm, device, save_results = True)

    data = {"loss_history": loss_history,
            "val_loss_history": val_loss_history,
            "train_x_norm": train_x_norm,
            "train_y_norm": train_y_norm,
            "test_x_norm": test_x_norm,
            "test_y_norm": test_y_norm}
    
    plots(model, data, device)
    



