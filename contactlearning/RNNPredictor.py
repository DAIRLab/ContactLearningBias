import numpy as np
import matplotlib.pyplot as plt
import torch
import os
# import pdb
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import torch.nn as nn
import torch.nn.functional as F

class CubeTossDataset(Dataset):
  def __init__(self, X, Y):
    self.len = X.shape[1]
    self.X = torch.FloatTensor(X)
    self.Y = torch.FloatTensor(Y)

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    return self.X[:,idx,:], self.Y[idx]


class RNNPredictor(nn.Module):

  def __init__(self, mode, input_size, hidden_size, output_size, dropout = 0.2):
    super(RNNPredictor, self).__init__()

    self.hidden_size = hidden_size
    self.mode = mode
    self.input_size = input_size
    self.output_size = output_size
    self.dropout = dropout

    if mode not in ["rnn", "lstm", "bilstm", "gru"]:
      raise ValueError("Choose a mode from - lstm/bisltm/rnn/gru")

    #Recurrent Layer
    if mode == "rnn":
      self.recurrent = nn.RNN(input_size = self.input_size, hidden_size = self.hidden_size,\
                                num_layers = 1, batch_first = False, dropout = self.dropout, bidirectional = False)
    elif mode == "lstm":
      self.recurrent = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size,\
                                num_layers = 1, batch_first = False, dropout = self.dropout, bidirectional = False)
    elif mode == "gru":
      self.recurrent = nn.GRU(input_size = self.input_size, hidden_size = self.hidden_size,\
                                num_layers = 1, batch_first = False, dropout = self.dropout, bidirectional = False)
    elif mode == "bilstm":
      self.recurrent = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size,\
                                num_layers = 1, batch_first = False, dropout = self.dropout, bidirectional = True)
    
    #Fully-connected Layers
    self.fc1 = nn.Linear(self.hidden_size, int(self.hidden_size/2))
    self.fc2 = nn.Linear(int(self.hidden_size/2), self.output_size)

  def forward(self, input_):

    if (self.mode == "rnn" or self.mode == "gru"):
      (_,hidden_output) = self.recurrent(input_)
    elif self.mode == "lstm":
      _, (hidden_output, _) = self.recurrent(input_)
    elif self.mode == "bilstm":
      _, (hidden_output_con, _) = self.recurrent(input_)
      hidden_output1 = hidden_output_con[0]
      hidden_output2 = hidden_output_con[1]
      hidden_output = (hidden_output1 + hidden_output2)[None,:,:]
      
    x = F.relu(self.fc1(hidden_output.squeeze(0)))
    x = self.fc2(x)

    return x

class MLPPredictor(nn.Module):

  def __init__(self, input_size, output_size, hidden_units = [256, 256, 256, 256]):
    super(MLPPredictor, self).__init__()

    self.output_size = output_size
    self.hidden_units = hidden_units
    self.input_size = input_size

    self.hidden_layers = nn.ModuleList()

    input_dim = self.input_size

    # a fully connected layer with len(hidden_units) layers
    for i in range(len(hidden_units)):
      self.hidden_layers += [nn.Linear(input_dim, self.hidden_units[i])]
      self.hidden_layers += [nn.ReLU()]
      input_dim = self.hidden_units[i]

    #output layer
    self.output = nn.Linear(input_dim, self.output_size)

  def forward(self, x):

    x = torch.squeeze(x, 0)
    for layer in self.hidden_layers:
      x = layer(x)

    return self.output(x)

if __name__ == "__main__":
	print("RNNPredictor.py executed!")