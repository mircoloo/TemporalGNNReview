import torch
import os
import sys
from models.DGDNN.Model.dgdnn import DGDNN
import models.DGDNN.Model.GGD
import models.DGDNN.Model.CatAttn
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
        
    

  



def main() -> None:
    #filter_stocks_from_timeperiod(['AAPL', 'MSFT'], ['2012-01-01', '2025-04-01'], Path("/home/mbisoffi/tests/TemporalGNNReview/code/data/datasets/hist_prices/America_Stocks"))
    ...
if __name__ == "__main__":
    main()





