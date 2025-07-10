import numpy as np
import pandas as pd
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader # For batching graph data
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
from pathlib import Path
import yaml
from typing import Final
#sklearn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm

# objective functions
from utils import theta_regularizer, neighbor_distance_regularizer, load_model
from dataset_utils import filter_stocks_from_timeperiod, retrieve_company_list

#Path managment
import os
import sys

from models.DGDNN.Data.geometric_dataset_gen import MyDataset as MyGeometricDataset
from models.DGDNN.Data.dataset_gen import MyDataset 



def main() -> None:

    # assigned the root project path    
    PROJECT_PATH: Final[Path] = Path(__file__).parent

    # load configuration paths 
    NASDAQ_CONFIG_PATH: Final[Path] = PROJECT_PATH / Path("configs/nasdaq_config_original.yaml")
    MODELS_CONFIG_PATH: Final[Path] = PROJECT_PATH / Path("configs/models_config.yaml")
    MODELS_WEIGHTS_PATH: Final[Path] = PROJECT_PATH  / Path("models/weights")

    # load data path
    hist_price_stocks_path = PROJECT_PATH / Path("data/datasets/hist_prices/America_Stocks")
    graph_dest_path =  PROJECT_PATH / Path("data/datasets/graph")
    tickers_csv_path = PROJECT_PATH / Path("data/tickers/NASDAQ.csv")
      
    
    # Load the YAML configutarion files
    with open(NASDAQ_CONFIG_PATH, 'r') as f:
        dataset_yaml = yaml.load(f, Loader=yaml.SafeLoader)
    with open(MODELS_CONFIG_PATH, 'r') as f:
        models_yaml = yaml.load(f, Loader=yaml.SafeLoader)
    

    # assigne the dataset variables
    dataset_param: dict = dataset_yaml['dataset_params']
    market:str = dataset_param['market']
    train_sedate: list[str] = dataset_param['train_sedate']
    val_sedate: list[str] = dataset_param['val_sedate']
    test_sedate: list[str] = dataset_param['test_sedate']
    window_size: int = dataset_param['window_size']
    use_fast_approximation: bool = dataset_param['use_fast_approximation']
    dataset_type = ['Train', 'Validation', 'Test']

    # Retrieve the company list and filter it for suitable dates 
    company_list = retrieve_company_list(tickers_csv_path)
    total_time_period: list[str] = [min(train_sedate + val_sedate + test_sedate), max(train_sedate + val_sedate + test_sedate)] 
    filtered_company_list = filter_stocks_from_timeperiod(company_list, market,total_time_period, hist_price_stocks_path)

    print(f"Company list length={len(company_list)}")
    print(f"Company list filtered length={len(filtered_company_list)}")


    #Build or retrieve the datasets if already present
    print("-"*5, "Building train dataset..." , "-"*5)
    train_dataset = MyGeometricDataset(hist_price_stocks_path, graph_dest_path, market, filtered_company_list, train_sedate[0], train_sedate[1], window_size, dataset_type[0], use_fast_approximation) 
    print("-"*5, "Building validation dataset..." , "-"*5)
    validation_dataset = MyGeometricDataset(hist_price_stocks_path, graph_dest_path, market, filtered_company_list, val_sedate[0], val_sedate[1], window_size, dataset_type[1], use_fast_approximation)
    print("-"*5, "Building test dataset..." , "-"*5)
    test_dataset = MyGeometricDataset(hist_price_stocks_path, graph_dest_path, market, filtered_company_list, test_sedate[0], test_sedate[1], window_size, dataset_type[2], use_fast_approximation)

    print("========== Train ==========")
    print(f"Number of snapshots graph of the dataset: {len(train_dataset)}")
    print(f"Graph snapshot structure: {train_dataset[0]}")
    print(f"Number of features: {int(train_dataset[0].x.shape[1] / window_size)}")
    print(f"Number of nodes: {len(train_dataset[0].x)}")
    print(f"Number of target labels: {len(train_dataset[0].y)}")

    print("========== Validation ==========")

    print(f"Number of snapshots graph of the dataset: {len(validation_dataset)}")
    print(f"Graph snapshot structure: {validation_dataset[0]}")
    print(f"Number of features: {int(validation_dataset[0].x.shape[1] / window_size)}")
    print(f"Number of nodes: {len(validation_dataset[0].x)}")
    print(f"Number of target labels: {len(validation_dataset[0].y)}")

    print("========== Test ==========")

    print(f"Number of snapshots graph of the dataset: {len(test_dataset)}")
    print(f"Graph snapshot structure: {test_dataset[0]}")
    print(f"Number of features: {int(test_dataset[0].x.shape[1] / window_size)}")
    print(f"Number of nodes: {len(test_dataset[0].x)}")
    print(f"Number of target labels: {len(test_dataset[0].y)}")
    

    # ================= TRAIN SECTION ================
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)  
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    

    # Build the model
    layers = 6
    num_nodes = train_dataset[0].x.shape[0]
    expansion_step = 7
    num_heads = 2
    active_layers = [True, False, False, False, False, False]
    timestamp = window_size | 19  #window size
    classes = 1
    diffusion_size = [5*timestamp, 31*timestamp, 28*timestamp, 24*timestamp, 20*timestamp, 16*timestamp, 12*timestamp]
    emb_size = [5 + 31, 64, 28 + 64, 50,
                24 + 50, 38, 20 + 38, 24,
                16 + 24, 12, 12+12, 10]  

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # get the model parameters from the configuration file 
    dataset_param: dict = models_yaml['DGDNN']
    num_nodes = train_dataset[0].x.shape[0]
    # load the model
    DGDNN = load_model('DGDNN')
    
    model_DGDNN = DGDNN(
        diffusion_size= diffusion_size, #list(dataset_param['diffusion_size']),
        embedding_size=  emb_size, #list(dataset_param['embedding_size']),
        classes=1,
        layers= layers, #dataset_param['layers'],
        num_nodes=num_nodes, #dataset_param['num_nodes'],
        expansion_step= expansion_step, #dataset_param['expansion_step'],
        num_heads=dataset_param['num_heads'],
        active= active_layers, #dataset_param['active_layers'],
        timestamp=timestamp #dataset_param['timestamp']
    ).to(device)
    print(f"Number of parameters={sum([p.numel() for p in model_DGDNN.parameters()])}")

    # ================ Train ===============
    optimizer = optim.Adam(model_DGDNN.parameters(), lr=2e-4, weight_decay=1.5e-5)
    criterion = nn.BCEWithLogitsLoss() # For binary classification (output is a single logit)
    
    num_epochs = 100
    model_DGDNN.train()

    losses: list[float] = []
    for epoch in  tqdm(range(num_epochs+1)): 
        train_loss = .0 #loss for each epoch
        for train_sample in train_loader: 
            if train_sample.x.shape[-1] != 5 * timestamp:
                print( f"Found {train_sample.x.shape}, SKIP" ) 
                continue
            #print("Putting the sample to the device ")
            train_sample = train_sample.to(device)
            optimizer.zero_grad()
            
            
            A = to_dense_adj(train_sample.edge_index, 
                                batch=train_sample.batch, # Important for correct batch processing if batch_size > 1
                                edge_attr=train_sample.edge_attr, 
                                max_num_nodes=num_nodes).squeeze(0) # Squeeze for batch_size=1
            
            C = train_sample.y.unsqueeze(dim=1).float()
            # Forward pass
            # DGDNN expects X: [num_nodes, features], A: [num_nodes, num_nodes]
            #print("Feeding the sample")
            outputs = model_DGDNN(train_sample.x, A) # Output shape: [num_nodes, classes]

            loss = criterion(outputs, C) - 0.0029 * neighbor_distance_regularizer(model_DGDNN.theta) + theta_regularizer(model_DGDNN.theta)
            #print(f"{loss=}")
            losses.append(loss.cpu().item())
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            #loss.step()
            train_loss += loss.item()

            # Validation step
        if epoch%2 == 0:
            val_loss =.0 #Reset validation loss
            val_acc = .0 #Reset validation accuracy
            val_f1 = .0 #Reset f1 score
            with torch.no_grad():
                for val_sample in validation_loader:
                    if val_sample.x.shape[-1] != 5 * timestamp:
                        print( f"Found {val_sample.x.shape}, SKIP" ) 
                        continue
                    val_sample.to(device)
                    A = to_dense_adj(val_sample.edge_index, 
                                    batch=val_sample.batch, # Important for correct batch processing if batch_size > 1
                                    edge_attr=val_sample.edge_attr, 
                                    max_num_nodes=num_nodes).squeeze(0) # Squeeze for batch_size=1
                    out = model_DGDNN(val_sample.x, A)

                    y = val_sample.y.detach().cpu() # Obtain the y labels
                    preds = (out > 0).float().detach().cpu().squeeze() #get the predicted logits

                    val_loss += criterion(out, val_sample.y.unsqueeze(1).float()).item()
                    val_acc += accuracy_score(y, preds)
                    val_f1 += f1_score(y, preds, zero_division=0)
                    #val_mcc += matthews_corrcoef() 

            print(f"Epoch {epoch}/{num_epochs},  train_loss={train_loss:.4f}, val_loss={(val_loss/len(validation_loader)):.4f}, val_acc={(val_acc/len(validation_loader)):.4f}, val_f1={(val_f1/len(validation_loader)):.4f}")


    print("Training finished.")

    # Saving the model paramters
    torch.save(model_DGDNN.state_dict(), MODELS_WEIGHTS_PATH / Path("model_DGDNN_weights.pth"))
    


if __name__ == '__main__':
    main()

