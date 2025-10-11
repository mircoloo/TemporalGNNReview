from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score
import torch
import os
import sys

from models.DGDNN.dgdnn import DGDNN
from models.GraphWaveNet.gwnet import gwnet
from models.DARNN.DARNN import MultiStockDARNN
from models.hyperstockgat.training.models.base_models import NCModel
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np





        
def load_model(model_name: str):
    match model_name:
        case "DGDNN":
            return DGDNN
        case "GraphWaveNet":
            return gwnet
        case "DARNN":
            return MultiStockDARNN
        case "hyperstockgat": #to implement later
             return NCModel
        case "DTML":
            from models.DTML.DTML import DTML
            return DTML
        case _:  # default case for any other model name
            raise ValueError(f"Unknown model name: {model_name}")
        
    


class Logger():
    def __init__(self, log_path):
        pass






def main() -> None:
    #filter_stocks_from_timeperiod(['AAPL', 'MSFT'], ['2012-01-01', '2025-04-01'], Path("/home/mbisoffi/tests/TemporalGNNReview/code/data/datasets/hist_prices/America_Stocks"))
    ...
if __name__ == "__main__":
    main()




