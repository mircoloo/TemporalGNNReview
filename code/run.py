import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm


# --- Local Imports ---
# Ensure these paths are correct relative to your project structure.
# You might need to adjust them if your project layout is different.
try:
    from models.DGDNN.Data.geometric_dataset_gen import MyDataset as MyGeometricDataset
    from utils import (log_test_results, neighbor_distance_regularizer,
                       theta_regularizer, load_model)
    from dataset_utils import filter_stocks_from_timeperiod, retrieve_company_list
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure your Python path is set up correctly and the necessary files exist.")
    sys.exit(1)


def main(args: argparse.Namespace) -> None:
    """
    Main function to run the DGDNN model training and evaluation pipeline.
    Accepts command-line arguments to specify the market.
    """
    # ------------------ 1. SETUP PATHS AND CONFIGS BASED ON ARGS ------------------
    PROJECT_PATH = Path(__file__).parent.resolve()
    
    market_name = args.market.lower()
    print(f"🚀 Starting process for market: {market_name.upper()}")

    # Dynamically load configuration and data paths
    MARKET_CONFIG_PATH = PROJECT_PATH / f"configs/{market_name}_config.yaml"
    MODELS_WEIGHTS_PATH = PROJECT_PATH / "models/weights"
    
    # Create weights directory if it doesn't exist
    MODELS_WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Assuming data is in a fixed relative location
    hist_price_stocks_path = PROJECT_PATH / f"data/datasets/hist_prices/{market_name.upper()}"
    graph_dest_path = PROJECT_PATH / "data/datasets/graph"
    tickers_csv_path = PROJECT_PATH / f"data/tickers/{market_name.upper()}.csv"

    # Load market-specific configuration file
    try:
        with open(MARKET_CONFIG_PATH, 'r') as f:
            config_yaml = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found at {MARKET_CONFIG_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error parsing YAML config: {repr(e)}")
        sys.exit(1)

    # ------------------ 2. ASSIGN VARIABLES FROM CONFIG ------------------
    dataset_param = config_yaml['dataset_params']
    model_param = config_yaml['model_params']
    train_param = config_yaml['training_params']

    market = dataset_param['market']
    train_sedate = dataset_param['train_sedate']
    val_sedate = dataset_param['val_sedate']
    test_sedate = dataset_param['test_sedate']
    window_size = dataset_param['window_size']
    use_fast_approximation = dataset_param['use_fast_approximation']
    
    # ------------------ 3. LOAD AND PREPARE DATASET ------------------
    company_list = retrieve_company_list(tickers_csv_path)
    total_time_period = [min(train_sedate + val_sedate + test_sedate), max(train_sedate + val_sedate + test_sedate)]
    filtered_company_list = filter_stocks_from_timeperiod(company_list, market, total_time_period, hist_price_stocks_path)

    print(f"Original company list length: {len(company_list)}")
    print(f"Filtered company list length: {len(filtered_company_list)}")

    # Build or retrieve the datasets
    print("-" * 5, "Building train dataset...", "-" * 5)
    train_dataset = MyGeometricDataset(hist_price_stocks_path, graph_dest_path, market, filtered_company_list, train_sedate[0], train_sedate[1], window_size, 'Train', use_fast_approximation)
    print("-" * 5, "Building validation dataset...", "-" * 5)
    validation_dataset = MyGeometricDataset(hist_price_stocks_path, graph_dest_path, market, filtered_company_list, val_sedate[0], val_sedate[1], window_size, 'Validation', use_fast_approximation)
    print("-" * 5, "Building test dataset...", "-" * 5)
    test_dataset = MyGeometricDataset(hist_price_stocks_path, graph_dest_path, market, filtered_company_list, test_sedate[0], test_sedate[1], window_size, 'Test', use_fast_approximation)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    num_nodes = train_dataset[0].x.shape[0]
    print(f"Number of nodes (stocks): {num_nodes}")

    # ------------------ 4. BUILD THE MODEL FROM CONFIG PARAMS ------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    DGDNN = load_model('DGDNN')
    
    model_DGDNN = DGDNN(
        diffusion_size=model_param['diffusion_size'],
        embedding_size=model_param['embedding_size'],
        classes=1,
        layers=model_param['layers'],
        num_nodes=num_nodes,
        expansion_step=model_param['expansion_step'],
        num_heads=model_param['num_heads'],
        active=model_param['active_layers'],
        timestamp=window_size
    ).to(device)
    
    print(f"Model parameters: {sum([p.numel() for p in model_DGDNN.parameters()]):,}")

    # ------------------ 5. TRAIN THE MODEL ------------------
    optimizer = optim.Adam(model_DGDNN.parameters(), lr=train_param['learning_rate'], weight_decay=train_param['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = train_param['epochs']
    alpha = train_param.get('neighbour_radius_coeff', 0.0) # Get alpha, default to 0 if not present
    
    model_DGDNN.train()
    
    print("\n" + "="*10 + " TRAINING " + "="*10)
    for epoch in tqdm(range(num_epochs + 1), desc="Epochs"):
        train_loss = 0.0
        for train_sample in train_loader:
            if train_sample.x.shape[-1] != 5 * window_size:
                print(f"Warning: Skipping sample with incorrect shape: {train_sample.x.shape}")
                continue
            
            train_sample = train_sample.to(device)
            optimizer.zero_grad()
            
            A = to_dense_adj(train_sample.edge_index, batch=train_sample.batch, edge_attr=train_sample.edge_attr, max_num_nodes=num_nodes).squeeze(0)
            C = train_sample.y.unsqueeze(dim=1).float()
            
            outputs = model_DGDNN(train_sample.x, A)
            
            # Loss from paper: L_CE - alpha * L_neighbor_dist + L_theta_reg
            loss = criterion(outputs, C) \
                   - alpha * neighbor_distance_regularizer(model_DGDNN.theta) \
                   + theta_regularizer(model_DGDNN.theta)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validation step ---
        if epoch % 30 == 0:
            val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
            model_DGDNN.eval() # Switch to evaluation mode
            with torch.no_grad():
                for val_sample in validation_loader:
                    if val_sample.x.shape[-1] != 5 * window_size: continue
                    val_sample = val_sample.to(device)
                    
                    A = to_dense_adj(val_sample.edge_index, batch=val_sample.batch, edge_attr=val_sample.edge_attr, max_num_nodes=num_nodes).squeeze(0)
                    out = model_DGDNN(val_sample.x, A)
                    
                    y_true = val_sample.y.detach().cpu()
                    y_pred = (out > 0).float().detach().cpu().squeeze()
                    
                    val_loss += criterion(out, val_sample.y.unsqueeze(1).float()).item()
                    val_acc += accuracy_score(y_true, y_pred)
                    val_f1 += f1_score(y_true, y_pred, zero_division=0)
            
            model_DGDNN.train() # Switch back to training mode
            avg_val_loss = val_loss / len(validation_loader)
            avg_val_acc = val_acc / len(validation_loader)
            avg_val_f1 = val_f1 / len(validation_loader)
            print(f"Epoch {epoch}/{num_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, Val F1: {avg_val_f1:.4f}")

    print("✅ Training finished.")

    # ------------------ 6. TEST THE MODEL ------------------
    print("\n" + "="*10 + " TESTING " + "="*10)
    model_DGDNN.eval()
    all_logits = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)

    with torch.no_grad():
        for test_sample in test_loader:
            if test_sample.x.shape[-1] != 5 * window_size: continue
            test_sample = test_sample.to(device)
            
            A = to_dense_adj(test_sample.edge_index, batch=test_sample.batch, edge_attr=test_sample.edge_attr, max_num_nodes=num_nodes).squeeze(0)
            
            out = model_DGDNN(test_sample.x, A)
            
            all_logits = torch.cat((all_logits, out.squeeze()), dim=0)
            all_labels = torch.cat((all_labels, test_sample.y), dim=0)

    # Calculate final metrics
    labels_cpu = all_labels.detach().cpu()
    preds_cpu = (all_logits > 0).float().detach().cpu()
    
    test_acc = accuracy_score(labels_cpu, preds_cpu)
    test_f1 = f1_score(labels_cpu, preds_cpu)
    test_mcc = matthews_corrcoef(labels_cpu, preds_cpu)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Test MCC: {test_mcc:.4f}")
    
    # ------------------ 7. SAVE RESULTS AND MODEL ------------------
    output_model_path = MODELS_WEIGHTS_PATH / f"model_DGDNN_{market_name}_weights.pth"
    torch.save(model_DGDNN.state_dict(), output_model_path)
    print(f"💾 Model weights saved to: {output_model_path}")
    
    log_file_name = f"{market_name}_run.log"
    log_path = PROJECT_PATH / 'logs'
    log_test_results(log_file_name, log_path, epochs=num_epochs, test_acc=test_acc, test_f1=test_f1, test_mcc=test_mcc)
    print(f"📄 Log file saved to: {log_path / log_file_name}")


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="Train and evaluate DGDNN model for a specific stock market.")

    # Add the required --market argument
    parser.add_argument(
        '--market',
        type=str,
        required=True,
        help="The stock market to process (e.g., 'nasdaq', 'nyse', 'sse'). This name is used to find the corresponding config and tickers file."
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function, passing the parsed arguments
    main(args)