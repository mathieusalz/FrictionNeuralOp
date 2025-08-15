# Neural Operator Library

This library contains an implementation of a simple naive neural operator as well as a fourier neural operator. There are utilities function for setting up models, training models, hyperparameter tuning models, saving models, and creating static or interactive plots.


## Installation

In order to start using this library, first clone this repository. Then, make sure you have all the package requirements, contained in ```requirements.txt```. These requirements can be automatically pip installed using the following command:

```bash
pip install -r requirements.txt
```

Preferably, these packages should be installed on a virtual environment.

If using this package on the SCITAS cluster, all the relevant modules to load are contained in ```train_FNO.sh```. 

## Project Structure

The current project structure is as follows:

```
NeuralOperator/
├── dashboards/
│   ├── compare_models_friction.py
│   └── compare_models_pulse.py
├── data/
│   ├── friction_data/
│   └── pulse_data/
│   └── signal_data/
├── data/
│   ├── data_prep_pulse.ipynb
│   └── data_prep_signals.ipynb
│   └── recover_transfer.ipynb
│   └── utils.py
├── results/
├── FNO.py
├── hyper_utils.py
├── hyperparameter_tuning.py
├── INO.py
├── interactive_utils.py
├── linear_pulse.ipynb
├── plot_utils.py
├── preprocess_utils.py
├── ReadMe.md
├── requirements.txt
├── save_utils.py
├── train_FNO.sh
├── train_utils.py
└── train.py
```

```FNO.py``` and ```INO.py``` are implementations of the FNO and simple integral neural operator as mentioned before. the rest of the files in the top layer are mostly utility scripts, however, the most important file to understand is ```train.py```, as it hosts all the functionality for easily training FNOs on the friction data.

## Training FNOs on Linear Pulse Data

The notebook ```linear_pulse.ipynb``` gives an example of how to train an FNO on linear pulse data. In addition, there are interactive plots to show the effect of hyperparameters on the performance of the FNO, as well as the performance of the FNO on the training data, testing data, and out of distribution data. This notebook should give a good understanding of how to setup simple FNOs and how to use some of the training, saving, and plotting utilities.

## Training FNOs on Friction Data

In order to train an FNO, all the relevant hyperparameters have to be configured using a dictionary. This dictionary is set up slightly differently depending on whether a single FNO learns on the full friction data, or whether two FNOs are used, one focusing on the healing data and the other on the state data of the friction.

Let's start with the single model:

### Single Friction FNO

The top structure of the dictionary is as follows:

```python
single_config = {'model_type': 'single',
                 'data': data_config,
                "train": train_config,
                "lift": lift_config,
                "fno": fno_config,
                "decode": decode_config}
```

As can be seen, we have to specify that the FNO is of type 'single'. Next, we need to supply separate configurations for the data, the training, the lifting layer, the FNO layers, and the decode layers. Let's go through these one by one.

#### Data Configuration

An example configuration is the following:

```python
data_config = {"x_path": "data/friction_data/features_AgingLaw_v2.csv",
               "y_path": "data/friction_data/targets_AgingLaw_v2.csv",
               'train_samples': 700,
               'log_norm': False,
               'state_norm': True,
               'heal_norm': True}
```

We must specify the file paths for the features and target data, how many samples we want to use for the train/test split, and finally the normalizations we want to use for the data. 
Due to the nature of the data, we need to apply normalizations to it. The log normalization is as follows: $x_{norm} = log(log(\frac{1}{x}))$. This works for the whole domain of the data. The state normalization is of the form $sqrt(log(1e8 * x))$ while the heal normalization is a binary mask which checks for values of $1e-8$, setting those values to 1 and all other (larger) numbers to 0. 
when training a single FNO, these can be mixed and matched, however, if only one is used, it should be the log_norm.

#### Train Configuration

The basic training hyperparameters:

```python
train_config = {'lr': 1e-3,
                'decay': 0.9,
                'decay_steps': 1000,
                'save_results': True,
                'epochs': 100}
```
#### Lift and Decode Configuration

The lift and decode configurations have basically the same configuration. Both can either be set up as a single 1D convolution to either lift or project down the input, or as Neural Networks.

The 1D convoltution set-up:

```python
lift_config = {"NN" : False,
               "act": "GELU"
               }
```

The Neural network set-up:

```python
decode_config = {"NN": True,
                 "NN_params": {"width": 32,
                               "depth": 1},
                 "act": "SiLU"
                 }
```

#### FNO Configuration

An example below: 

```python
fno_config = {"mode": 16,
              "blocks": 4,
              "act": "GELU",
              "width": 32,
              "padding": 9,
              "coord_features": True,
              "adaptive": True,
              }
```
The coord_features parameter adds an extra channel with uniformly spaced points between 0 and 1, in order to integrate the coordinates into the data. The adaptive parameter is for using adaptive activation functions.

### Dual FNO Configuration

A dual FNO configuration is not very different from a single FNO configuration, except for the fact that a full configuration must be provided for both the healing FNO and the state FNO:

```python
dual_config = {"model_type": 'dual',
          "data": data_config,
          "train": train_config,
          "pretrain": {"load": False,
                       "lr": 1e-3,
                       "epochs": 25,
                       "save_results": True},
          "heal": {"fno" : fno_config,
                   "lift" : lift_config,
                   "decode": decode_config},
          "state": {"fno" : dc(fno_config),
                   "lift" : dc(lift_config),
                   "decode": dc(decode_config)}
          }
```

In addition, we can also specify if we want the state FNO to be pretrained, as we have data which only contains the state part of friction. If pretraining is not used, it can be omitted from the configuration dictionnary.

## Training an FNO

Once the configuration dictionary has been created, the hard part is done. All that is left is to run these lines (already set-up in ```train.py```):

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
check_config(single_config)
data = prepare_data(single_config, device)
model = model_setup(single_config, data, device)
train_model(single_config, model, data, device)
plots(model, data, device)
```
The function ```check_config()``` will make sure that there are no issues with the configuration dictionnary before anything happens. Then, the data is automatically preprocessed, the model is setup, trained, and plots for loss history as well as inference on the training and testing data are automatically created.


## To-Do

While this library has extensive functionality, there are still things which should be implemented or fixed:

- fixing ```hyperparamter_tuning.py``` to math with the new configuration setup and additional hyperparameters
- in the training configuration (```train_utils.py```), adding gradient-clipping, mixed resolution and noisy training as options (with the features already implemented)
- when creating and saving a model, there should be a folder containing the configuration dictionary, the plots, and the model weights for better tracking and reproducibility (```save_utils.py```, ```plot_utils.py```)
- all the utility scripts should be put into a single folder to make the repository tidier
- use ```sphinx``` to create proper and detailed documentation


