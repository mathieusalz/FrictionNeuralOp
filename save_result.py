import equinox as eqx
import pickle
import jax

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
        - model parameters using Equinox
        - loss_history, val_history, data, model_class, model_hparams using pickle
        """

        # Extract hyperparameters automatically from model
        model_hparams = {
            "in_channels": self.model.in_channels,
            "out_channels": self.model.out_channels,
            "modes": self.model.modes,
            "width": self.model.width,
            "activation": self.model.activation,
            "n_blocks": self.model.n_blocks}
        
        for key, val in model_hparams.items():
            if callable(val):
                model_hparams[key] = val.__name__

        # Construct filename prefix safely
        filename_prefix = f"{self.data}_{self.model_class.__name__}_"
        for key in model_hparams:
            val = model_hparams[key]
            # Convert function names to strings if needed
            if callable(val):
                val = val.__name__
            filename_prefix += f"{key}{val}_"

        model_filename = f"trained_models/{filename_prefix}_model.eqx"

        # Save model parameters
        eqx.tree_serialise_leaves(model_filename, self.model)

        other_data = {
            "loss_history": self.loss_history,
            "val_history": self.val_history,
            "data": self.data,
            "model_class": self.model.__class__,
            "model_hparams": model_hparams,
            "model_filename": model_filename,
        }

        with open(f"trained_models/{filename_prefix}_data.pkl", "wb") as f:
            pickle.dump(other_data, f)

    @classmethod
    def load(cls, filename_prefix):
        """
        Loads Model_Result without requiring a model template by using saved hyperparameters.

        Args:
            filename_prefix (str): prefix for saved files

        Returns:
            Model_Result instance with loaded data and reconstructed model.
        """
        with open(f"trained_models/{filename_prefix}_data.pkl", "rb") as f:
            other_data = pickle.load(f)

        model_hparams = other_data["model_hparams"]
        model_filename = other_data["model_filename"]
        model_class = other_data["model_class"]

        activation = model_hparams['activation']
        if activation is not None:
            if activation == 'relu':
                model_hparams['activation'] = jax.nn.relu
            elif activation == 'tanh':
                model_hparams['activation'] = jax.nn.tanh

        # Reconstruct model template
        model_template = model_class(**model_hparams)

        # Load model parameters
        model = eqx.tree_deserialise_leaves(model_filename, model_template)

        return cls(
            model=model,
            loss_history=other_data["loss_history"],
            val_history=other_data["val_history"],
            data=other_data["data"],
        )
