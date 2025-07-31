import torch
import pickle
import os

class Model_Result:
    def __init__(self, model, loss_history, val_history, data):
        self.model = model
        self.model_class = model.__class__
        self.loss_history = loss_history
        self.val_history = val_history
        self.data = data

    def save(self):
        """
        Saves:
        - model parameters using torch
        - loss_history, val_history, data, model_class, model_hparams using pickle
        """

        # Extract hyperparameters automatically from model
        model_hparams = {
            "in_channels": self.model.in_channels,
            "out_channels": self.model.out_channels,
            "modes": self.model.modes,
            "width": self.model.width,
            "activation": self.model.activation,
            "n_blocks": self.model.n_blocks,
            "padding": self.model.padding
        }

        for key, val in model_hparams.items():
            if callable(val):
                model_hparams[key] = val.__name__

        # Construct filename prefix safely
        filename_prefix = f"{self.data}_{self.model_class.__name__}_"
        for key in model_hparams:
            val = model_hparams[key]
            filename_prefix += f"{key}{val}_"

        os.makedirs("trained_models", exist_ok=True)
        model_filename = f"trained_models/{filename_prefix}_model.pt"

        # Save model parameters using torch
        torch.save(self.model.state_dict(), model_filename)

        other_data = {
            "loss_history": self.loss_history,
            "val_history": self.val_history,
            "data": self.data,
            "model_class_name": self.model_class.__name__,
            "model_hparams": model_hparams,
            "model_filename": model_filename,
        }

        with open(f"trained_models/{filename_prefix}_data.pkl", "wb") as f:
            pickle.dump(other_data, f)

    @classmethod
    def load(cls, filename_prefix, model_class):
        """
        Loads Model_Result by using saved hyperparameters and model class.

        Args:
            filename_prefix (str): prefix for saved files
            model_class (class): class of the model to reconstruct

        Returns:
            Model_Result instance with loaded data and reconstructed model.
        """
        with open(f"trained_models/{filename_prefix}_data.pkl", "rb") as f:
            other_data = pickle.load(f)

        model_hparams = other_data["model_hparams"]
        model_filename = other_data["model_filename"]

        activation = model_hparams['activation']
        if activation is not None:
            if activation == 'relu':
                model_hparams['activation'] = torch.nn.functional.relu
            elif activation == 'tanh':
                model_hparams['activation'] = torch.tanh
            elif activation == 'gelu':
                model_hparams['activation'] = torch.nn.functional.gelu

        # Reconstruct model template
        model = model_class(**model_hparams)

        # Load model parameters
        model.load_state_dict(torch.load(model_filename, map_location='cpu'))

        return cls(
            model=model,
            loss_history=other_data["loss_history"],
            val_history=other_data["val_history"],
            data=other_data["data"],
        )
