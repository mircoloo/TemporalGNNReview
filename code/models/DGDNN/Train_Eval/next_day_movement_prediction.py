import torch
import torch.nn.functional as F
from torch_geometric.logging import log
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
from matplotlib import cm
from matplotlib import axes
import seaborn as sns
import sklearn.preprocessing as skp
from sklearn.metrics import matthews_corrcoef, f1_score
import os 
import sys
from utils import theta_regularizer, neighbor_distance_regularizer

model_path = os.path.abspath('/Users/mirco/Documents/Tesi/code/models/DGDNN/Model')
data_path = os.path.abspath('/Users/mirco/Documents/Tesi/code/models/DGDNN/Data')
if model_path not in sys.path:
    sys.path.insert(0, model_path)
if data_path not in sys.path:
    sys.path.insert(0, data_path)

from dataset_gen import MyDataset
from dgdnn import DGDNN

# Configure the device for running the model on GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configure the default variables // # these can be tuned // # examples
sedate = ['2013-01-01', '2014-12-31']  # these can be tuned
val_sedate = ['2015-01-01', '2015-06-30'] # these can be tuned
test_sedate = ['2015-07-01', '2017-12-31'] # these can be tuned
market = ['NASDAQ', 'NYSE', 'SSE'] # can be changed
dataset_type = ['Train', 'Validation', 'Test']
com_path = ['/content/drive/MyDrive/Raw_Data/Stock_Markets/NYSE_NASDAQ/NASDAQ.csv',
            '/content/drive/MyDrive/Raw_Data/Stock_Markets/NYSE_NASDAQ/NYSE.csv',
            '/content/drive/MyDrive/Raw_Data/Stock_Markets/NYSE_NASDAQ/NYSE_missing.csv']
com_path = ['/Users/mirco/Documents/Tesi/code/data/datasets/NASDAQ.csv',
            '/Users/mirco/Documents/Tesi/code/data/datasets/NYSE.csv',
            '/Users/mirco/Documents/Tesi/code/data/datasets/NYSE_missing.csv'
            ]

des = '/content/drive/MyDrive/Raw_Data/Stock_Markets/NYSE_NASDAQ/raw_stock_data/stocks_indicators/data'
des = "/Users/mirco/Documents/Tesi/code/data/datasets/graph"
directory = "/content/drive/MyDrive/Raw_Data/Stock_Markets/NYSE_NASDAQ/raw_stock_data/stocks_indicators/data/google_finance"
directory = "/Users/mirco/Documents/Tesi/code/data/datasets/America_Stocks"
window = 19
NASDAQ_com_list = []
NYSE_com_list = []
NYSE_missing_list = []
com_list = [NASDAQ_com_list, NYSE_com_list, NYSE_missing_list] #Ticker lists


for idx, path in enumerate(com_path): #Per each ticker folder get the path and index
    with open(path) as f:
        file = csv.reader(f)
        for line in file:
            com_list[idx].append(line[0])  # append first element of line if each line is a list
NYSE_com_list = [com for com in NYSE_com_list if com not in NYSE_missing_list] #Filter the com on NYSE since they are missing 
fast_approx = False # True for fast approximation and implementation
# Generate datasets
print("-"*5, "Building train dataset..." , "-"*5)
#Market: 0:Nasdaq 1:NYSE 2:SSE
 #                      root: str, desti: str, market: str, comlist: List[str], start: str, end: str, window: int, dataset_type: str, fast_approx
train_dataset = MyDataset(directory, des, market[0], NASDAQ_com_list, sedate[0], sedate[1], window, dataset_type[0], fast_approx) 
print("-"*5, "Building validation dataset..." , "-"*5)
validation_dataset = MyDataset(directory, des, market[0], NASDAQ_com_list, val_sedate[0], val_sedate[1], window, dataset_type[1], fast_approx)
print("-"*5, "Building test dataset..." , "-"*5)
test_dataset = MyDataset(directory, des, market[0], NASDAQ_com_list, test_sedate[0], test_sedate[1], window, dataset_type[2], fast_approx)


print("-" * 5, "Defining the model..." , "-"*5)
####  Define model
layers = 6
num_nodes = 1026 
expansion_step = 7
num_heads = 2
active_layers = [True, False, False, False, False, False]
timestamp = 19  #window
classes = 2
diffusion_size = [5*timestamp, 31*timestamp, 28*timestamp, 24*timestamp, 20*timestamp, 16*timestamp, 12*timestamp]
emb_size = [5 + 31, 64, 28 + 64, 50,
            24 + 50, 38, 20 + 38, 24,
            16 + 24, 12, 12+12, 10]  

model = DGDNN(
    diffusion_size=diffusion_size,
    embedding_size=emb_size,
    classes=1,
    layers=layers,
    num_nodes=num_nodes,
    expansion_step=expansion_step,
    num_heads=num_heads,
    active=active_layers,
    timestamp=timestamp
)

# Pass model GPU
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1.5e-5)


# Define training process & validation process & testing process

epochs = 6000
model.reset_parameters()

# Training
for epoch in range(epochs):
    model.train()
    objective_total = 0
    correct = 0
    total = 0

    for sample in train_dataset: #Recommend to update every sample, full batch training can be time-consuming
        X = sample['X'].to(device)  # node feature tensor
        A = sample['A'].to(device)  # adjacency tensor
        C = sample['Y'].long()
        C = C.to(device)  # label vector
        optimizer.zero_grad()
        out = model(X, A)
        objective = F.cross_entropy(out, C) - 0.0029 * neighbor_distance_regularizer(model.theta) + theta_regularizer(model.theta)
        # to fast implement can omit the two regularization terms + theta_regularizer(model.theta) - 0.0029 * neighbor_distance_regularizer(model.theta)
        objective.backward()
        optimizer.step()
        objective_total += objective.item()

    # If performance progress of the model is required
        out = out.argmax(dim=1)
        correct += int((out == C).sum()).item()
        total += C.shape[0]
        if epoch % 1 == 0:
          print(f"Epoch {epoch}: loss={objective_total:.4f}, acc={correct / total:.4f}")

# Validation
model.eval()

# Define evaluation metrics
# ACC, MCC, and F1
acc = 0
f1 = 0
mcc = 0

for idx, sample in enumerate(validation_dataset):

    X = sample['X']  # node feature tensor
    A = sample['A']  # adjacency tensor
    C = sample['Y']  # label vector
    out = model(X, A).argmax(dim=1)

    acc += int((out == C).sum())
    f1 += f1_score(C, out.cpu().numpy())
    mcc += matthews_corrcoef(C, out.cpu().numpy())

print(acc / (len(validation_dataset) * C.shape[0]))
print(f1 / len(validation_dataset))
print(mcc/ len(validation_dataset))

# Test

acc = 0
f1 = 0
mcc = 0

for idx, sample in enumerate(test_dataset):
    X = sample['X']  # node feature tensor
    A = sample['A']  # adjacency tensor
    C = sample['Y']  # label vector
    out = model(X, A).argmax(dim=1)

    acc += int((out == C).sum())
    f1 += f1_score(C, out.cpu().numpy())
    mcc += matthews_corrcoef(C, out.cpu().numpy())

print(acc / (len(test_dataset) * C.shape[0]))
print(f1 / len(test_dataset))
print(mcc / len(test_dataset))
