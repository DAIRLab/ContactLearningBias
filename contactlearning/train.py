#first
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pdb
import json
import sys
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import torch.nn as nn
import torch.nn.functional as F

from RNNPredictor import RNNPredictor, CubeTossDataset
import learning_utils
import eval_utils 

import warnings
import argparse

parser = argparse.ArgumentParser(description="train a recurrent model for learning contact dynamics")

parser.add_argument("--stiffness", default = 2500, type = int, choices = [2500, 300, 100], \
		 help = "the stiffness setting to use")

parser.add_argument("--tw", default = 16, type = int, \
		 help = "the time-window for recurrency")

parser.add_argument("--train_tosses", default = 500, type = int, \
		 help = "the num of tosses for training data")

parser.add_argument("--test_tosses", default = 500, type = int, \
		 help = "the num of tosses for test data")

parser.add_argument("--normalize", action = "store_true", \
		 help = "if provided, it will normalize the input to the network")

parser.add_argument("--proportional_test", action = "store_true", \
		 help = "if provided, it will use test data proporational to training data size")

parser.add_argument("--batch_size", default = 64, type = int, \
		 help = "batch-size to use for training")

parser.add_argument("--recurrent_mode", default = "lstm", type = str, choices = ["mlp", "lstm", "bilstm", "gru"], \
		 help = "type of recurrent variant to use")

parser.add_argument("--predict_mode", default = "vel", type = str, choices = ["vel", "dvel"], \
		 help = "part of state to use for training the model to predict")

parser.add_argument("--lr", default = 1e-4, type = float, \
		 help = "learning rate to use for training")

parser.add_argument("--hidden_size", default = 256, type = int, \
		 help = "hidden layer size to use for the network, in powers of 2")

parser.add_argument("--toss_start", default = 1, type = int, \
		 help = "for training it will uses tosses from [toss_start, toss_start + train_tosses]")

parser.add_argument("--epoch_start", default = 0, type = int, \
		 help = "the epoch number to start training at")

parser.add_argument("--num", type = int, \
		 help = "specify which training iteration this corresponds to")

parser.add_argument("--resume", action = "store_true", \
		 help = "if training is to resumes from a previously set point")

parser.add_argument("--weighted_loss", help="uses a weighted loss if provided", action="store_true")

parser.add_argument("--rot_weight", default = 1, type = float, \
		 help = "the weight assigned to rotation component")

parser.add_argument("--pos_weight", default = 1, type = float, \
		 help = "the weight assigned to position component")

parser.add_argument("--verbose", help="increase output verbosity", action="store_true")

parser.add_argument("--weight_decay", type = float, \
		 help = "weight decay to use for regularization")

parser.add_argument("--theta_noise", type = float, \
		 help = "rotational noise")

parser.add_argument("--com_noise", type = float, \
		 help = "noise for c.o.m. position")
args = parser.parse_args()

#stiffness
STIFFNESS_VAL = args.stiffness

#perturb
PERTURB_WIDTH = 10

#Recurrency time-window
tw = args.tw

#amount of data
NUM_TOSSES = args.train_tosses
TEST_TOSSES = args.test_tosses
TOSS_START = args.toss_start

#scaling
NORMALIZE_ = args.normalize
COMMON_TEST = not args.proportional_test

#weighted loss
WEIGHTED_LOSS = args.weighted_loss
WEIGHTS = [args.pos_weight, args.rot_weight]

#Tunable hyperparameters
BATCH_SIZE = args.batch_size
RECURRENT_MODE = args.recurrent_mode #one of ["lstm", "bilstm", "rnn", "gru"]
PRED_MODE = "vel"
LEARNING_RATE = args.lr
HIDDEN_SIZE = args.hidden_size
WEIGHT_DECAY = args.weight_decay


#Fixed hyperparameters
OUTPUT_SIZE_DICT = {"vel": 6}
INPUT_SIZE = 13
OUTPUT_SIZE = OUTPUT_SIZE_DICT[PRED_MODE]

#parameters for resuming training
RESUME = args.resume
EPOCH_START = args.epoch_start
LAST_SAVE_EPOCH = -1
BEST_VAL_LOSS = 1e4
BEST_TRAIN_LOSS = 1e4

#training iteration
training_iteration = args.num

#Noise parameters
THETA_NOISE = 2
COM_NOISE = 0.002

DATA_PATH = "data"
MODELS_PATH = "models"

(X_train, y_train), (X_val, y_val), (X_test, y_test) = \
	learning_utils.getTrainingData(DATA_PATH, stiffness = STIFFNESS_VAL,perturb_width = PERTURB_WIDTH, \
		num_train_tosses = NUM_TOSSES, num_test_tosses = TEST_TOSSES, \
		common_test = COMMON_TEST, tw = tw, num_start = TOSS_START, predict_mode = PRED_MODE, theta_noise = THETA_NOISE, com_noise = COM_NOISE)

mean_, std_ = learning_utils.getScaling(X_train, normalize_ = NORMALIZE_)
print("mean_: ", mean_, "std_: ", std_)

train_dataset = CubeTossDataset(X_train.transpose((1,0,2)),y_train)
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)

val_dataset = CubeTossDataset(X_val.transpose((1,0,2)),y_val)
val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)

test_dataset = CubeTossDataset(X_test.transpose((1,0,2)),y_test)
test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device type: ", device)

MODEL_NAME = f"{STIFFNESS_VAL}-{PERTURB_WIDTH}-{RECURRENT_MODE}-{tw}-{HIDDEN_SIZE}-{LEARNING_RATE:.0e}-{PRED_MODE}-{NUM_TOSSES}tosses"

if WEIGHT_DECAY is not None:
	MODEL_NAME = f"{MODEL_NAME}-{WEIGHT_DECAY:.0e}"
if training_iteration is not None:
	MODEL_NAME = f"{training_iteration:02d}-{MODEL_NAME}"
if WEIGHTED_LOSS:
	MODEL_NAME = f"wloss-{MODEL_NAME}"
if NORMALIZE_:
	MODEL_NAME = f"norm-{MODEL_NAME}"
print("MODEL NAME: ", MODEL_NAME)

weight_decay = 0 if WEIGHT_DECAY is None else WEIGHT_DECAY
print("using weight decay: ", weight_decay)
# warnings.filterwarnings("ignore", category=UserWarning)
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	warnings.warn("Model definition dropout", UserWarning)
	if RECURRENT_MODE != "mlp":
		model = RNNPredictor(RECURRENT_MODE, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, \
				output_size=OUTPUT_SIZE)
	else:
		model = MLPPredictor(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, hidden_units = [HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE])
		tw = 1
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = weight_decay)

ROOT_LOG_DIR = "Logs"
TENSORBOARD_DIR = MODEL_NAME # Sub-Directory for storing this specific experiment's logs
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	warnings.warn("Logger Numpy deprecated version", FutureWarning)
	logger = SummaryWriter(os.path.join(ROOT_LOG_DIR, TENSORBOARD_DIR))

TRAINING_DONE = False
if RESUME:
	model.load_state_dict(torch.load(os.path.join(MODELS_PATH + "/train_states", MODEL_NAME + "-train") + ".pt", map_location=device))
	with open(f'{MODELS_PATH}/train_states/{MODEL_NAME}-state.json') as f:
		state_data = json.load(f)
	EPOCH_START = state_data["last_epoch"]
	LAST_SAVE_EPOCH = state_data["last_save_epoch"]
	BEST_VAL_LOSS = state_data['best_val_loss']
	BEST_TRAIN_LOSS = state_data['best_train_loss']
	TRAINING_DONE = state_data["done"]

if not TRAINING_DONE:
	if RESUME:
		print(f"Restarting training process from epoch: {EPOCH_START}")
	TRAIN_LOSS, VAL_LOSS, _ = learning_utils.train_predictor(model, train_dataloader, val_dataloader, loss_function, optimizer, epoch_start = EPOCH_START, pred_mode = PRED_MODE, \
	                num_epochs=300, scale_mean = (mean_), scale_std = (std_), \
	                logger = logger, print_every=200, model_name = MODEL_NAME, WAIT = 20, device = device,\
	                last_save_epoch = LAST_SAVE_EPOCH, best_val_loss = BEST_VAL_LOSS, best_train_loss = BEST_TRAIN_LOSS, weighted_loss = WEIGHTED_LOSS, weights = WEIGHTS)
else:
	print(f"Training process is done, last save epoch: {LAST_SAVE_EPOCH}\nExiting code!")
	sys.exit(0)

model = eval_utils.loadModel(MODELS_PATH, model, model_name = MODEL_NAME)

TEST_LOSS, _, _ = learning_utils.evaluate_predictor(model, test_dataloader, loss_function, pred_mode = PRED_MODE, scale_mean = mean_, scale_std = std_, \
				recurrent = True, device = device, weighted_loss = WEIGHTED_LOSS, weights = WEIGHTS)

data_path = os.path.join("/home/mihir/DAIR/compact_data", str(STIFFNESS_VAL), str(PERTURB_WIDTH))
TRAIN_ROLLOUT_LOSS, TRAIN_ROLLOUT_ROT_ERR, TRAIN_ROLLOUT_POS_ERR = eval_utils.evaluateRollout(os.path.join(data_path, "mujoco_sim"), model, loss_function, num_tosses = min(500,NUM_TOSSES), toss_start = TOSS_START, pred_mode = PRED_MODE,\
													 isTest = False, tw = tw, mean_ = mean_, std_ =  std_, weighted_loss = WEIGHTED_LOSS, weights = WEIGHTS, theta_noise = THETA_NOISE, com_noise = COM_NOISE, synthetic_vel = True)

TEST_ROLLOUT_LOSS, TEST_ROLLOUT_ROT_ERR, TEST_ROLLOUT_POS_ERR = eval_utils.evaluateRollout(os.path.join(data_path, "mujoco_sim"), model, loss_function, num_tosses = 500, pred_mode = PRED_MODE,\
													 isTest = True, tw = tw, mean_ = mean_, std_ =  std_, weighted_loss = WEIGHTED_LOSS, weights = WEIGHTS, theta_noise = THETA_NOISE, com_noise = COM_NOISE, synthetic_vel = True)

TRAIN_SINGLESTEP_LOSS, TRAIN_SINGLESTEP_ROT_ERR, TRAIN_SINGLESTEP_POS_ERR = eval_utils.evaluateSinglestep(os.path.join(data_path, "mujoco_sim"), \
													model, loss_function, num_tosses = min(500,NUM_TOSSES), toss_start = TOSS_START, pred_mode = PRED_MODE, tw = tw, mean_ = mean_, std_ =  std_, theta_noise = THETA_NOISE, com_noise = COM_NOISE, synthetic_vel = True)

TEST_SINGLESTEP_LOSS, TEST_SINGLESTEP_ROT_ERR, TEST_SINGLESTEP_POS_ERR = eval_utils.evaluateSinglestep(os.path.join(data_path, "mujoco_sim"), \
													model, loss_function, num_tosses = 500, pred_mode = PRED_MODE, isTest = True, tw = tw, mean_ = mean_, std_ =  std_, theta_noise = THETA_NOISE, com_noise = COM_NOISE, synthetic_vel = True)

STATS_SAVE_PATH = f'Results/{MODEL_NAME}-result'
STATS_SAVE_DATA = {"TRAIN_ROLLOUT_LOSS": TRAIN_ROLLOUT_LOSS, "TRAIN_ROLLOUT_ROT_ERR": TRAIN_ROLLOUT_ROT_ERR, "TRAIN_ROLLOUT_POS_ERR": TRAIN_ROLLOUT_POS_ERR,\
					"TEST_ROLLOUT_LOSS": TEST_ROLLOUT_LOSS, "TEST_ROLLOUT_ROT_ERR": TEST_ROLLOUT_ROT_ERR, "TEST_ROLLOUT_POS_ERR": TEST_ROLLOUT_POS_ERR, \
					"TRAIN_SINGLESTEP_LOSS": TRAIN_SINGLESTEP_LOSS, "TRAIN_SINGLESTEP_ROT_ERR": TRAIN_SINGLESTEP_ROT_ERR, "TRAIN_SINGLESTEP_POS_ERR": TRAIN_SINGLESTEP_POS_ERR, \
					"TEST_SINGLESTEP_LOSS": TEST_SINGLESTEP_LOSS, "TEST_SINGLESTEP_ROT_ERR": TEST_SINGLESTEP_ROT_ERR, "TEST_SINGLESTEP_POS_ERR": TEST_SINGLESTEP_POS_ERR, \
					"TRAIN_LOSS": TRAIN_LOSS, "VAL_LOSS": VAL_LOSS, "TEST_LOSS": TEST_LOSS}

with open(STATS_SAVE_PATH+".json", "w") as write_file:
	json.dump(STATS_SAVE_DATA, write_file, indent = 2)

print("Successfully Executed")

