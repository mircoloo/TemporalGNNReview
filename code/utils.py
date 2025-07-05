import torch
import os
import sys
from models.DGDNN.Model.dgdnn import DGDNN
import models.DGDNN.Model.GGD
import models.DGDNN.Model.CatAttn


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


def train_model_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss =.0

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr)  # or appropriate inputs
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def create_test_report(model_name, epochs,dest, acc, f1, mcc, prediction_balance):
    full_path = os.path.join(os.getcwd(), f"{dest}/{model_name}_{epochs}.txt")
    if os.path.exists( full_path ):
        print("File already exists...")
        return
    with open(full_path, 'w+') as f:
        f.write(f"{model_name}_{epochs}\n")
        f.write(f"EPOCHS={epochs}\n")
        f.write(f"ACCURACY={acc}\n")
        f.write(f"F1Score={f1}\n")
        f.write(f"MCC={mcc}")

        
def load_model(model_name: str):
    match model_name:
        case "DGDNN":
            return DGDNN
    