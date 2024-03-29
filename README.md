# Challenges in Learning Stiff Contact Dynamics

This repository contains numerical experiments that accompany the following publication:

Mihir Parmar*, Mathew Halm*, and Michael Posa. "Fundamental Challenges in Deep Learning for Stiff Contact Dynamics," IROS 2021. 

## Requirements
* Python 3.6.9 or higher
* 16GB RAM
* Linux (tested on Ubuntu 18.04 LTS)

GPU is not necessary, although it will provide significant speedup for for the training process.

## Python Dependencies
* numpy 1.19.5
* matplotlib 3.3.2
* PyTorch 1.7.0
* scipy 1.5.4
* sklearn 0.23.2
* argparse 1.1

## MuJoCo Installation and Generating Data
A precomputed set of trajectories can be downloaded from [here](https://drive.google.com/drive/folders/1AX2J17WYvDL4rSj16imHIUV2RZ5ysnWx?usp=sharing). However, the *data-scripts* directory contains the scripts to generate additional data and visualize die roll trajectories.

To generate more data, install [MuJoCo 200](https://www.roboti.us/index.html) at `~/.mujoco/mjpro200`, copy your license key to `~/.mujoco/mjkey.txt` and move the [`cube_toss.xml`](data-scripts/cube_toss.xml) that defines the system model to `/.mujoco/model/`.

To simulate the die roll system in MuJoCo and generate trajectories, use the desired contact settings by changing the stiffness and dampping values under the `solref` tag part of the xml script and then run:
```
python3 generate.py
```
This will result in 11,000 trajectories being stored at *contactlearning/data/<stiffness_value_used>/*

To visalize any trajectory using MuJoCo's rendering, run:
```
python3 visualize.py <path to trajectory>
```

## Training and Evaluation
Under *contactlearning*, [`RNNPredictor.py`](contactlearning/RNNPredictor.py) defines the model architecture used in the experiments and supports MLP, LSTM, GRU, and BiLSTM architectures.

To train a model on the set of generated trajectories with specific training settings, run:
```
python3 train.py <training-settings>
```
### Training Settings and hyperparameters
The possible arguments under `<training-settings>` include:
* `--stiffness <value>`, the stiffness value from {2500, 300, 100} corresponding to which data is to used (default = 2500)
* `--train_tosses <value>`, number of training trajectories upto 10000 (default = 500)
* `--tw <value>`, history-length (default 16)
* `--normalize`, normalize the input data
* `--batch_size <value>`, batch-size to use during training (default = 64)
* `--recurrent_mode <value>`, DNN architecture from {mlp, lstm, gru, bilstm} to use (default = lstm)
* `--lr <value>`, learning-rate value to use with Adam Optimizer (default = 1e-4)
* `--hidden_size <value>`, width of the hidden-layer of the RNN (default = 256)
* `--weight_decay <value>`, weight-decay to use for regularization (default = 0)

Checkpoints updated after every epoch as well as the final trained models are stored in *contactlearning/models/*.

### Logging
`tensorboard` logs from training are stored in *contactlearning/Logs*. Training and validation loss curves can be visualized at `localhost:6006` using:
```
tensorboard --logdir=contactlearning/Logs
```

### Evaluation and results
[`eval_utils.py`](contactlearning/eval_utils.py) includes helper methods used in `train.py` to evaluate trained models. 
Both training and evaluation results are stored in a `.json` file at *contactlearning/Results*
