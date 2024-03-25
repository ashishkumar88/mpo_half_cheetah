# MPO RL Algorithm Implementation in Python

This project is an implementation of the Maximum a Posteriori Policy Optimization (MPO) algorithm, a state-of-the-art reinforcement learning method, using Python, Torch and Torch RL. This implementation only work  for continuous control tasks. 


## Installation

Before training the model, you need to install the following dependencies:

```bash
sudo apt install -y libx11-dev
sudo apt-get install -y libglew-dev
sudo apt-get install -y patchelf
```

Then, you can install the required Python packages using the following command:

```bash
pip install -r requirements.txt
```

## Directory Structure

The project directory is structured as follows:

- `config`: Contains the configuration files for the model.
- experiment.py: The main script to train and test the model.
- trainer.py: The script that contains the training and testing logic.
- model.py: The script that contains the model architecture.
- utils.py: The script that contains utility functions.
- requirements.txt: The file that contains the required Python packages.
- mpo.py : The script that contains the MPO algorithm implementation.
- logger.py : The script that contains the logger implementation.
- LICENSE.md: The license file for the project.

## Training

To train the model, you can run the following command:

```bash
python experiment.py --config <absolute path to config_file or relative path wrt root directory>
```

The configuration file is a yaml file that contains the hyperparameters for the model. You can find an example of the configuration file in the `config` directory.

## Testing

To test the model, you can run the following command:

```bash
python experiment.py --config <absolute path to config_file or relative path wrt root directory> --test --checkpoint_dir <absolute path to the checkpoint directory>
```

## License
This program is licensed under GPL 3, license [here](LICENSE.md).