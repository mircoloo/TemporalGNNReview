from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score
import torch
import os
import sys

from models.DGDNN.Model.dgdnn import DGDNN
import models.DGDNN.Model.ggd
import models.DGDNN.Model.catattn
from models.GraphWaveNet.gwnet import gwnet

from models.DARNN.DARNN import DARNN
#from models.HyperStockGAT.training.models.base_models import NCModel
from pathlib import Path
from datetime import datetime
import pandas as pd

from models.HyperStockGAT.training.models.base_models import NCModel 

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
        case "DARNN":
            return DARNN
        case "HyperStockGraph":
            return NCModel
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

def process_test_results(y_hat, y_true):
    """
    Print and return evaluation metrics for test results (binary classification).
    """
    # Convert raw outputs to binary predictions
    

    acc = accuracy_score(y_true, y_hat)
    precision = precision_score(y_true, y_hat, zero_division=0)
    recall = recall_score(y_true, y_hat, zero_division=0)
    f1 = f1_score(y_true, y_hat, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_hat)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test MCC: {mcc:.4f}")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mcc": mcc
    }

def remove_unshaped_samples(orig_dataset, n_nodes:int,sequence_length:int, features:int):
    expected_second_dimension = features * sequence_length
    filtered_samples = []

# Iterate through each sample in your dataset
    for i, sample in enumerate(original_dataset):
        # Ensure the sample has an 'x' attribute and it's a tensor/array with at least 2 dimensions
        if hasattr(sample, 'x') and isinstance(sample.x, (torch.Tensor, np.ndarray)) and sample.x.dim() >= 2:
            # Check the condition
            if sample.x.shape[1] == expected_second_dimension:
                filtered_samples.append(sample)
            else:
                print(f"Removing sample at index {i} because its x.shape[1] ({sample.x.shape[1]}) "
                    f"does not match {expected_second_dimension}")
        else:
            print(f"Skipping sample at index {i} as it doesn't have a valid 'x' attribute.")




def main() -> None:
    #filter_stocks_from_timeperiod(['AAPL', 'MSFT'], ['2012-01-01', '2025-04-01'], Path("/home/mbisoffi/tests/TemporalGNNReview/code/data/datasets/hist_prices/America_Stocks"))
    ...
if __name__ == "__main__":
    main()





