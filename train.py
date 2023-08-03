import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from plot import plot
from definitions import FruitRipenessDetector, evaluate, train

# Continue Training Model
paramys = torch.load("model_ripeness_detector_bs64_lr0.001_epoch125")
model = FruitRipenessDetector()
model.load_state_dict(paramys)


train_dataset = torch.load('train_dataset_v2.pth')
val_dataset = torch.load('val_dataset_v2.pth')

train(model, train_dataset, val_dataset, batch_size=64, print_stat=True, num_epochs=25, current_epoch=125)