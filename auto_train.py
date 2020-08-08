import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from math import cos, pi

import skopt
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.metrics import mean_squared_error
from  skopt.plots import plot_convergence

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from torch.utils.tensorboard import SummaryWriter

import read_preprocess
from models import DNN, dataset_csv

class structured_train():
    def __init__(self, file_path, target, split):
        self.file_path = file_path
        self.target = target
        self.split = split

        x_train, x_val, y_train, y_val, tgt_type = read_preprocess.read_and_preprocess(file_path, target, split)
        self.x_train = torch.from_numpy(x_train.to_numpy()).float()
        self.y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())

        self.x_val = torch.from_numpy(x_val.to_numpy()).float()
        self.y_val = torch.squeeze(torch.from_numpy(y_val.to_numpy()).float())
