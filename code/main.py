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

# objective functions
from utils import theta_regularizer, neighbor_distance_regularizer, load_model


#Path managment
import os
import sys

from models.DGDNN.Data.geometric_dataset_gen import MyDataset as MyGeometricDataset
from models.DGDNN.Data.dataset_gen import MyDataset 



def main() -> None:
    
    PROJECT_PATH: Final[Path] = Path(__file__).parent

    NASDAQ_CONFIG_PATH: Final[Path] = PROJECT_PATH / Path("configs/nasdaq_config.yaml")
    MODELS_CONFIG_PATH: Final[Path] = PROJECT_PATH / Path("configs/models_config.yaml")
    MODELS_WEIGHTS_PATH: Final[Path] = PROJECT_PATH  / Path("models/weights")
    
    print(NASDAQ_CONFIG_PATH)
    
    

    with open(NASDAQ_CONFIG_PATH, 'r') as f:
        dataset_yaml = yaml.load(f, Loader=yaml.SafeLoader)
    with open(MODELS_CONFIG_PATH, 'r') as f:
        models_yaml = yaml.load(f, Loader=yaml.SafeLoader)
    
    

    dataset_param: dict = dataset_yaml['dataset_params']
    market:str = dataset_param['market']
    train_sedate: list[str] = dataset_param['train_sedate']
    val_sedate: list[str] = dataset_param['val_sedate']
    test_sedate: list[str] = dataset_param['test_sedate']
    window_size: int = dataset_param['window_size']
    use_fast_approximation: bool = dataset_param['use_fast_approximation']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset_param: dict = models_yaml['DGDNN']

    DGDNN = load_model('DGDNN')
    model_DGDNN = DGDNN(
        diffusion_size=dataset_param['diffusion_size'],
        embedding_size=dataset_param['embedding_size'],
        classes=1,
        layers=dataset_param['layers'],
        num_nodes=dataset_param['num_nodes'],
        expansion_step=dataset_param['expansion_step'],
        num_heads=dataset_param['num_heads'],
        active=dataset_param['active_layers'],
        timestamp=dataset_param['timestamp']
).to(device)

    hist_price_stocks_path = PROJECT_PATH / Path("data/datasets/hist_prices/America_Stocks")
    graph_dest_path =  PROJECT_PATH / Path("data/datasets/graph")
    tickers_csv_path = PROJECT_PATH / Path("data/tickers/NASDAQ.csv")

    market = 'NASDAQ'

    with open(tickers_csv_path, 'r') as f:
        content = f.read()
        company_list = content.split()

    print(company_list)
    #company_list = company_list
    dataset_type = ['Train', 'Validation', 'Test']

    
    print("-"*5, "Building train dataset..." , "-"*5)
    train_dataset = MyGeometricDataset(hist_price_stocks_path, graph_dest_path, market, company_list, train_sedate[0], train_sedate[1], window_size, dataset_type[0], use_fast_approximation) 
    print("-"*5, "Building validation dataset..." , "-"*5)
    validation_dataset = MyGeometricDataset(hist_price_stocks_path, graph_dest_path, market, company_list, val_sedate[0], val_sedate[1], window_size, dataset_type[1], use_fast_approximation)
    print("-"*5, "Building test dataset..." , "-"*5)
    test_dataset = MyGeometricDataset(hist_price_stocks_path, graph_dest_path, market, company_list, test_sedate[0], test_sedate[1], window_size, dataset_type[2], use_fast_approximation)
#clear_output()

    ### TEMPORARY

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    print(len(train_loader))
    print(len(validation_loader))
    print(len(test_loader))


    torch.save(model_DGDNN.state_dict(), MODELS_WEIGHTS_PATH / Path("model_DGDNN_weights.pth"))
    

if __name__ == '__main__':
    main()

