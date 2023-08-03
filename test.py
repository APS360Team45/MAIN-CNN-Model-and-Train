import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from plot import plot
from definitions import FruitRipenessDetector, evaluate

TEST_ON_INDIVIDUAL_SETS = False

MODEL_PATH_1 = "full-train-and-model\\v1\model-weights\model_ripeness_detector_bs64_lr0.001_epoch60"
MODEL_PATH_2 = "full-train-and-model\\v1\model-weights\model_ripeness_detector_bs64_lr0.001_epoch30"
MODEL_PATH_3 = 'full-train-and-model\\v2\model-weights\model_ripeness_detector_bs64_lr0.001_epoch60'
MODEL_PATH_4 = 'full-train-and-model\\v2\model-weights\model_ripeness_detector_bs64_lr0.001_epoch87'
MODEL_PATH_5 = 'full-train-and-model\\v2\model-weights\model_ripeness_detector_bs64_lr0.001_epoch100'
MODEL_PATH_6 = 'full-train-and-model\\v2\model-weights\model_ripeness_detector_bs64_lr0.001_epoch150'
COMBINED_TEST_DATASET_PATH = "combined_test_dataset.pth"
TEST_DATASET_SADMAN_PATH = "test_dataset_extra(sadman).pth"
TEST_DATASET_ARTIN_PATH = "test_dataset_extra(artin).pth"

test_model = FruitRipenessDetector()
if not TEST_ON_INDIVIDUAL_SETS:

    test_dataset = torch.load(COMBINED_TEST_DATASET_PATH)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    paramys = torch.load(MODEL_PATH_3)
    test_model.load_state_dict(paramys)
    test_loss, test_accuracy = evaluate(test_model, test_loader, nn.MSELoss(), testing=True)
    print(f"Test Accuracy V2,Epoch 60: {test_accuracy*100}%")

    
    paramys = torch.load(MODEL_PATH_4)
    test_model.load_state_dict(paramys)
    test_loss, test_accuracy = evaluate(test_model, test_loader, nn.MSELoss(), testing=True)
    print(f"Test Accuracy V2,Epoch 87: {test_accuracy*100}%")


    paramys = torch.load(MODEL_PATH_5)
    test_model.load_state_dict(paramys)
    test_loss, test_accuracy = evaluate(test_model, test_loader, nn.MSELoss(), testing=True)
    print(f"Test Accuracy V2,Epoch 100: {test_accuracy*100}%")


    paramys = torch.load(MODEL_PATH_6)
    test_model.load_state_dict(paramys)
    test_loss, test_accuracy = evaluate(test_model, test_loader, nn.MSELoss(), testing=True)
    print(f"Test Accuracy V2,Epoch 150: {test_accuracy*100}%")


if TEST_ON_INDIVIDUAL_SETS:
    paramys = torch.load(MODEL_PATH_3)
    test_model.load_state_dict(paramys)

    test_dataset = torch.load(TEST_DATASET_SADMAN_PATH)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    test_loss, test_accuracy = evaluate(test_model, test_loader, nn.MSELoss(), testing=True)
    print(f"Test Accuracy V2,Epoch 60: (hand picked (sadman)): {test_accuracy*100}%")

    test_dataset = torch.load(TEST_DATASET_ARTIN_PATH)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    test_loss, test_accuracy = evaluate(test_model, test_loader, nn.MSELoss(), testing=True)
    print(f"Test Accuracy V2,Epoch 60: (hand picked (artin)): {test_accuracy*100}%")

    paramys = torch.load(MODEL_PATH_4)
    test_model.load_state_dict(paramys)

    test_dataset = torch.load(TEST_DATASET_SADMAN_PATH)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    test_loss, test_accuracy = evaluate(test_model, test_loader, nn.MSELoss(), testing=True)
    print(f"Test Accuracy V2,Epoch 87: (hand picked (sadman)): {test_accuracy*100}%")

    test_dataset = torch.load(TEST_DATASET_ARTIN_PATH)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    test_loss, test_accuracy = evaluate(test_model, test_loader, nn.MSELoss(), testing=True)
    print(f"Test Accuracy V2,Epoch 87: (hand picked (artin)): {test_accuracy*100}%")

    paramys = torch.load(MODEL_PATH_5)
    test_model.load_state_dict(paramys)

    test_dataset = torch.load(TEST_DATASET_SADMAN_PATH)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    test_loss, test_accuracy = evaluate(test_model, test_loader, nn.MSELoss(), testing=True)
    print(f"Test Accuracy V2,Epoch 100: (hand picked (sadman)): {test_accuracy*100}%")

    test_dataset = torch.load(TEST_DATASET_ARTIN_PATH)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    test_loss, test_accuracy = evaluate(test_model, test_loader, nn.MSELoss(), testing=True)
    print(f"Test Accuracy V2,Epoch 100: (hand picked (artin)): {test_accuracy*100}%")

    paramys = torch.load(MODEL_PATH_6)
    test_model.load_state_dict(paramys)

    test_dataset = torch.load(TEST_DATASET_SADMAN_PATH)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    test_loss, test_accuracy = evaluate(test_model, test_loader, nn.MSELoss(), testing=True)
    print(f"Test Accuracy V2,Epoch 150: (hand picked (sadman)): {test_accuracy*100}%")

    test_dataset = torch.load(TEST_DATASET_ARTIN_PATH)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    test_loss, test_accuracy = evaluate(test_model, test_loader, nn.MSELoss(), testing=True)
    print(f"Test Accuracy V2,Epoch 150: (hand picked (artin)): {test_accuracy*100}%")