import torch
import os
import sys

from models.DGDNN.Model.dgdnn import DGDNN
import models.DGDNN.Model.ggd
import models.DGDNN.Model.catattn
from models.GraphWaveNet.model import gwnet

from pathlib import Path
from datetime import datetime
import pandas as pd 

# Define optimizer and objective function
def theta_regularizer(theta):
    row_sums = torch.sum(theta, dim=-1)
    ones = torch.ones_like(row_sums)
    return torch.sum(torch.abs(row_sums - ones))

def neighbor_distance_regularizer(theta):
    box = torch.sum(theta, dim=-1)
    result = torch.zeros_like(theta)

    for idx, row in enumerate(theta):
        for i, j in enumerate(row):
            result[idx, i] = i * j

    result_sum = torch.sum(result, dim=1)
    return torch.sum(result / result_sum[:, None])


        
def load_model(model_name: str):
    match model_name:
        case "DGDNN":
            return DGDNN
        case "GraphWaveNet":
            return gwnet
        case _:  # default case for any other model name
            raise ValueError(f"Unknown model name: {model_name}")
        
    
def log_test_results(filename: Path | str, path_to_log: str | Path, **kwargs):
    path_to_log = Path(path_to_log) 
    if not path_to_log.exists():
        print(f"Folder {path_to_log} does not exists...")
        raise FileNotFoundError
    filepath = path_to_log / Path(filename)
    with open(filepath, 'a+') as f:
        for key,value in kwargs.items():
            f.write(f"{key}={value}\n")
    f.write("=======================\n")




def main() -> None:
    #filter_stocks_from_timeperiod(['AAPL', 'MSFT'], ['2012-01-01', '2025-04-01'], Path("/home/mbisoffi/tests/TemporalGNNReview/code/data/datasets/hist_prices/America_Stocks"))
    ...
if __name__ == "__main__":
    main()





