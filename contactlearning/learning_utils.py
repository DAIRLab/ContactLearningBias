import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import glob
import pdb
import sys
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline
import json

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import torch.nn as nn
import torch.nn.functional as F

def getTrainingData(PATH, stiffness = 2500, perturb_width = 10, \
		num_train_tosses = 500, num_test_tosses = 30, \
		common_test = True, tw = 16, num_start = 1, predict_mode = "vel"):

  X, y = getRecurrentDataset(os.path.join(PATH, str(stiffness), str(perturb_width), "mujoco_sim"), 
                              num_start, num_train_tosses, tw = tw, predict_mode = predict_mode)
  X_test, y_test = getRecurrentDataset(os.path.join(PATH, str(stiffness), str(perturb_width), "mujoco_sim"), 
                              num_start, num_test_tosses, tw = tw, isTest = True, predict_mode = predict_mode)
  slice_val = (num_train_tosses*80)
  X = X[:,:slice_val, :]
  y = y[:slice_val, :]
  print("For train: X shape = ", X.shape, ", Y shape = ", y.shape, "\n")
  print("For test: X shape = ", X_test.shape, ", Y shape = ", y_test.shape, "\n")

  if not common_test:
    X_train, X_test, y_train, y_test = train_test_split(X.transpose((1,0,2)), y, train_size = 0.9, random_state = 42 )
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = (2/9), random_state = 42 )
  else:
    X_train, X_val, y_train, y_val = train_test_split(X.transpose((1,0,2)), y, train_size = 0.9, random_state = 42 )
    X_test,y_test = X_test.transpose((1,0,2)),y_test
 
  
  print("Training: ",X_train.shape, "y_train: ", y_train.shape)
  print("Validation: ", X_val.shape, "y_val: ", y_val.shape)
  print("Test: ", X_test.shape, "y_test: ", y_test.shape)

  return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def getScaling(X, normalize_ = False):

	if normalize_:
		mean_ = np.mean(X,(0,1))
		std_ = np.std(X,(0,1))
	else:
		mean_ = np.array([0])
		std_ = np.array([1])

	return torch.from_numpy(mean_).float(), torch.from_numpy(std_).float()

def getDataLen(PATH, stiffness = 2500, \
		perturb_width = 10, num_tosses = 30, isTest = False, tw = 16, num_start = 1):
	
	FILE_PREFIX = os.path.join(PATH, str(stiffness), str(perturb_width), "mujoco_sim")
	if not isTest:
		NUM_START = num_start if (num_start+num_tosses < 9500) else (9500 - num_tosses)
	else:
		NUM_START = 9500
	NUM_END = NUM_START + num_tosses

	data_len = 0
	for i in range(NUM_START, NUM_END + 1):
		data_ = np.loadtxt(FILE_PREFIX + str(i) + ".csv", delimiter = ",").T
		data_len += len(data_) - tw
	print("In getDataLen: ",data_len)

	return data_len

def train_predictor(model, train_dataloader, val_dataloader, loss_function, optimizer, epoch_start = 0, pred_mode = "vel", num_epochs = 10, verbose = True, \
          recurrent = True, scale_mean = 0, scale_std = 1, logger = None, print_every = 2, model_name = "recurrent0", WAIT = 20, device = torch.device("cpu"), \
          last_save_epoch = -1, best_val_loss = 1e4, best_train_loss = 1e4, weighted_loss = False, weights = [1,1]):
  """
  For training the model
  """
  if weighted_loss:
    print("Using weighted loss with weights: ", weights)
  model.to(device)
  scale_mean, scale_std = scale_mean.to(device), scale_std.to(device)
  model.train()
  overall_step = 0
  models_path = "/home/mihir/DAIR/compact_data/models/"
  for epoch in range(epoch_start, num_epochs):
    print("Start of epoch: model.training: ", model.training)
    running_loss = 0
    positional_running_loss = 0
    rotational_running_loss = 0
    epoch_step = 0
    total = 0
    for (data, labels) in train_dataloader:
      input_ = np.transpose(data,(1,0,2))
      input_ = input_.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      output = model((input_ - scale_mean)/scale_std)
      if pred_mode == "full":
        loss = loss_function(output, labels[:,:])
      elif (pred_mode == "vel" or pred_mode == "dvel"):
        positional_loss = loss_function(output[:,:3], labels[:,7:10])
        rotational_loss = loss_function(output[:,3:], labels[:,10:])
        if weighted_loss:
          loss = (weights[0]*positional_loss) + (weights[1]*rotational_loss)
        else:
          loss = loss_function(output, labels[:, 7:])
      elif (pred_mode == "pos" or pred_mode == "dpos"):
        positional_loss = loss_function(output[:,:3], labels[:,:3])
        rotational_loss = loss_function(output[:,3:], labels[:,3:7])
        if weighted_loss:
          loss = (weights[0]*positional_loss) + (weights[1]*rotational_loss)
        else:
          loss = loss_function(output, labels[:,:7])
      elif (pred_mode == "dx"):
        positional_loss = loss_function(output[:,:3], labels[:,:3]) + loss_function(output[:,7:10], labels[:,7:10])
        rotational_loss = loss_function(output[:,3:7], labels[:,3:7]) + loss_function(output[:,10:], labels[:,10:])
        if weighted_loss:
          print("Weighted loss not supported in dx!")
          sys.exit(1)
        else:
          loss = loss_function(output, labels) 
      elif (pred_mode == "fs"):
        positional_loss = loss_function(output[:,:3], labels[:,:3])
        rotational_loss = loss_function(output[:,3:7], labels[:,3:7])
        if weighted_loss:
          print("Weighted loss not supported in dx!")
          sys.exit(1)
        else:
          loss = loss_function(output, labels[:,:7])            
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      positional_running_loss += positional_loss.item()
      rotational_running_loss += rotational_loss.item()
      total += labels.shape[0]

      if (epoch_step % print_every == 0):
        if verbose:
          print(" --- Epoch: %s Step: %s Loss: %s" %(epoch, epoch_step, running_loss/total))
      
      epoch_step += 1
      overall_step += 1

    logger.add_scalar("Loss/train", running_loss/total, epoch)
    logger.add_scalar("Positional_Loss/train", positional_running_loss/total, epoch)
    logger.add_scalar("Rotational_Loss/train", rotational_running_loss/total, epoch)

    val_loss, pos_val_loss, rot_val_loss = evaluate_predictor(model, val_dataloader, loss_function, pred_mode = pred_mode, scale_mean = scale_mean, scale_std = scale_std, recurrent=True,  device = device, \
                                                              weighted_loss = weighted_loss, weights = weights)
    logger.add_scalar("Loss/val", val_loss, epoch)
    logger.add_scalar("Positional_Loss/val", pos_val_loss, epoch)
    logger.add_scalar("Rotational_Loss/val", rot_val_loss, epoch)
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_train_loss = running_loss/total
      SAVE_PATH = models_path + model_name + ".pt"
      torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'mean_': scale_mean, 'std_': scale_std, 'optimizer_state_dict': optimizer.state_dict()}, SAVE_PATH)
      last_save_epoch = epoch
    if epoch - last_save_epoch > WAIT:
      break

    TRAIN_SAVE_PATH = models_path + "train_states/" + model_name + "-train.pt"
    torch.save(model.state_dict(), TRAIN_SAVE_PATH)
    state_save = {"last_epoch": epoch, 'last_save_epoch': last_save_epoch, \
                  'best_val_loss': best_val_loss, 'best_train_loss': best_train_loss, "done": False}
    with open(models_path + "train_states/" + model_name +"-state.json", "w") as write_file:
      json.dump(state_save, write_file)

    model.train()

  print("Training finished. Best val loss: ", best_val_loss, " @ epoch: ", last_save_epoch)
  state_save["done"] = True
  with open(models_path + "train_states/" + model_name +"-state.json", "w") as write_file:
      json.dump(state_save, write_file)
  return best_train_loss, best_val_loss, last_save_epoch

def evaluate_predictor(trained_model, test_dataloader, loss_function, pred_mode = "vel", scale_mean = 0, scale_std = 1, \
                        recurrent = True, device = torch.device("cpu"), weighted_loss = False, weights = [1,1]):
  """
  Validation of the trained model on test/val set.

  Inputs:
  trained_model: Object of the trained model
  test_dataloader: Object of the test/val dataloader
  loss_function: Object of the loss_function to be used

  Output:
  average loss over the dataset
  """
  if weighted_loss:
    print("Inside eval predictor -- using weighted loss with weights: ", weights)
  trained_model.to(device)
  scale_mean, scale_std = scale_mean.to(device), scale_std.to(device)
  trained_model.eval()
  running_loss = 0
  positional_running_loss = 0
  rotational_running_loss = 0
  overall_step = 0
  total = 0
  # weights = np.array([1, 1, 1, 100, 100, 100]).reshape(1,-1)
  with torch.no_grad():
    for (data, labels) in test_dataloader:
      input_ = np.transpose(data, (1,0,2))
      input_ = input_.to(device)
      # pdb.set_trace()
      labels = labels.to(device)
      output = trained_model((input_ - scale_mean)/scale_std)
      if pred_mode == "full":
        loss = loss_function(output, labels[:,:])
      elif (pred_mode == "vel" or pred_mode == "dvel"):
        positional_loss = loss_function(output[:,:3], labels[:,7:10])
        rotational_loss = loss_function(output[:,3:], labels[:,10:])
        if weighted_loss:
          loss = (weights[0]*positional_loss) + (weights[1]*rotational_loss)
        else:
          loss = loss_function(output, labels[:, 7:])
      elif (pred_mode == "pos" or pred_mode == "dpos"):
        positional_loss = loss_function(output[:,:3], labels[:,:3])
        rotational_loss = loss_function(output[:,3:], labels[:,3:7])
        if weighted_loss:
          loss = (weights[0]*positional_loss) + (weights[1]*rotational_loss)
        else:
          loss = loss_function(output, labels[:,:7])
      elif (pred_mode == "dx"):
        positional_loss = loss_function(output[:,:3], labels[:,:3]) + loss_function(output[:,7:10], labels[:,7:10])
        rotational_loss = loss_function(output[:,3:7], labels[:,3:7]) + loss_function(output[:,10:], labels[:,10:])
        if weighted_loss:
          print("Weighted loss not supported in dx!")
          sys.exit(1)
        else:
          loss = loss_function(output, labels)
      elif (pred_mode == "fs"):
        positional_loss = loss_function(output[:,:3], labels[:,:3])
        rotational_loss = loss_function(output[:,3:7], labels[:,3:7])
        if weighted_loss:
          print("Weighted loss not supported in dx!")
          sys.exit(1)
        else:
          loss = loss_function(output, labels[:,:7]) 
      running_loss += loss.item()
      positional_running_loss += positional_loss.item()
      rotational_running_loss += rotational_loss.item()
      overall_step += 1
      total += labels.shape[1]

  return running_loss/total, positional_running_loss/total, rotational_running_loss/total

def getRecurrentDataset(FILE_PREFIX, NUM_START, NUM_TOSSES, tw = 16, isTest = False, isEval = False, predict_mode = "vel"):
  
  X_list = []
  Y_list = []
  if not isEval:
    if not isTest:
      NUM_START = NUM_START if (NUM_START+NUM_TOSSES <= 10001) else (10001 - NUM_TOSSES)
    else:
      NUM_START = 10001
    print(f"Dataset starting from....{NUM_START}")
  NUM_END = NUM_START + NUM_TOSSES
  for k in range(NUM_START, NUM_END):
    data_ = np.loadtxt(FILE_PREFIX + str(k) + ".csv", delimiter = ",").T
    X = np.zeros((tw,0,13))
    Y = np.zeros((0,13))
    # print(str(i), "done")
    if predict_mode == "fs":
      X_sub = np.expand_dims(data_[0:tw], axis = 1)
      X = np.append(X,X_sub, axis = 1)
      Y_sub = (data_[-1].reshape(1,-1))
      Y = np.append(Y,Y_sub, axis = 0)
    else:
      for i in range(len(data_) - tw):
        X_sub = np.expand_dims(data_[i:i+tw], axis = 1)
        X = np.append(X, X_sub, axis = 1)
        if predict_mode == "vel":
          Y_sub = (data_[i+tw].reshape(1,-1))
        elif (predict_mode == "dvel" or predict_mode == "dpos" or predict_mode == "dx"):
          Y_sub = ((data_[i+tw] - data_[i+tw-1]).reshape(1,-1))
        Y = np.append(Y, Y_sub, axis = 0)
    X_list.append(X)
    Y_list.append(Y)

  return np.concatenate(X_list, axis = 1), np.concatenate(Y_list, axis = 0) 


if __name__ == "__main__":
	print("learning_utils.py executed!")