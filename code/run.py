import argparse
import sys
from pathlib import Path
import os

# Add the models directory to Python path
PROJECT_PATH = Path(__file__).parent.resolve()
sys.path.append(str(PROJECT_PATH))
sys.path.append(str(PROJECT_PATH / "models" / "DGDNN" / "Model"))

from model_runners.dgdnn_runner import DGDNNRunner
from model_runners.graphwavenet_runner import GraphWaveNetRunner
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, mean_squared_error, recall_score
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
from utils import load_model

torch.manual_seed(42)  # For reproducibility

# --- Local Imports ---
# Ensure these paths are correct relative to your project structure.
# You might need to adjust them if your project layout is different.
try:
    from dataset_utils import filter_stocks_from_timeperiod, retrieve_company_list
    from models.DGDNN.Data.geometric_dataset_gen import MyDataset as MyGeometricDataset
    from utils import (neighbor_distance_regularizer,
                       theta_regularizer, load_model, process_test_results)
except ImportError as e:
    print(f"Error importing local modules in run.py: {repr(e)}")
    print("Please ensure your Python path is set up correctly and the necessary files exist.")
except Exception as e:
    print(f"Exiting due to import error: {repr(e)}")

    sys.exit(1)



def main(args: argparse.Namespace) -> None:
    """
    Main function to run the DGDNN model training and evaluation pipeline.
    Accepts command-line arguments to specify the market.
    """
    # ------------------ 1. SETUP PATHS AND CONFIGS BASED ON ARGS ------------------
    PROJECT_PATH = Path(__file__).parent.resolve()
    sys.path.append(str(PROJECT_PATH))
    
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
    
    

    # ------------------ 5. TRAIN AND TEST THE MODEL USING THE RUNNER ------------------

    if args.model == 'dgdnn':
        DGDNN = load_model('DGDNN')
        model_param = model_param['DGDNN']
        model_DGDNN = DGDNN(
            diffusion_size=model_param['diffusion_size'],
            embedding_size=model_param['embedding_size'],
            embedding_hidden_size = model_param['embedding_hidden_size'],
            embedding_output_size = model_param['embedding_output_size'],
            raw_feature_size = model_param['raw_feature_size'],
            classes=1,
            layers=model_param['layers'],
            num_nodes=num_nodes,
            expansion_step=model_param['expansion_step'],
            num_heads=model_param['num_heads'],
            active=model_param['active_layers']
        ).to(device)
    
        print(f"Model parameters: {sum([p.numel() for p in model_DGDNN.parameters()]):,}")
        runner = DGDNNRunner(model_DGDNN, device)

        optimizer = optim.Adam(model_DGDNN.parameters(), lr=float(train_param['learning_rate']), weight_decay=float(train_param['weight_decay']))
        criterion = nn.BCEWithLogitsLoss()
        num_epochs = train_param['epochs']
        alpha = train_param.get('neighbour_radius_coeff', 0.0)
        runner.train(
            train_loader, validation_loader, optimizer, criterion, num_epochs,
            alpha, neighbor_distance_regularizer, theta_regularizer, window_size, num_nodes)
        print("✅ Training finished.")

        print("\n" + "="*10 + " TESTING " + "="*10)
        y_pred, y_true = runner.test(test_loader, window_size, num_nodes)
        process_test_results(y_pred, y_true)

        
    elif args.model == 'graphwavenet':
        print("GraphWaveNet model selected. Running training and evaluation pipeline.")
        GWN = load_model('GraphWaveNet')
        model_GWN = GWN(
            device=device,
            num_nodes=num_nodes,
            dropout=0.3,
            supports=None,
            gcn_bool=True,
            addaptadj=True,
            aptinit=None,
            in_dim=5,  # <-- number of features
            out_dim=1,                
            residual_channels=256, 
            dilation_channels=256,
            skip_channels=256,
            end_channels=512,
            kernel_size=1,
            blocks=1,
            layers=8
        ).to(device)
        runner = GraphWaveNetRunner(model_GWN, device)
        print(f"Model parameters: {sum([p.numel() for p in model_GWN.parameters()]):,}")

        optimizer = optim.Adam(model_GWN.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        num_epochs = train_param['epochs']

        runner.train(train_loader, validation_loader, optimizer, criterion, num_epochs, window_size, 5)
        print("✅ Training finished.")

        print("\n" + "="*10 + " TESTING " + "="*10)
        y_pred, y_true = runner.test(test_loader, window_size, 5)
        process_test_results(y_pred, y_true)
    elif args.model == 'darnn':
        from model_runners.darnn_runner import DARNNRunner
        DARNN = load_model('DARNN')
        model_DARNN = DARNN(
            num_nodes-1,
            64,
            64,
            T=window_size
        ).to(device)
        print(f"Model parameters: {sum([p.numel() for p in model_DARNN.parameters()]):,}")
        runner = DARNNRunner(model_DARNN, device)
        optimizer = optim.Adam(model_DARNN.parameters(), lr=float(train_param['learning_rate']), weight_decay=float(train_param['weight_decay']))
        criterion = nn.BCEWithLogitsLoss()
        runner.train(train_loader, validation_loader, optimizer, criterion, train_param['epochs'], seq_length=window_size)
        y_pred, y_true = runner.test(test_loader, seq_length=window_size, num_features=5)
        process_test_results(y_pred, y_true)    
    elif args.model == 'hyperstockgraph':
        from model_runners.hyperstockgraph_runner import HyperStockGraphRunner
        NCModel = load_model('HyperStockGraph')
        model_HSG = NCModel(
            in_dim=5,  # <-- number of features
            
        ).to(device)
        runner = HyperStockGraphRunner(model_HSG, device)
        print(f"Model parameters: {sum([p.numel() for p in model_HSG.parameters()]):,}")

        optimizer = optim.Adam(model_HSG.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        num_epochs = train_param['epochs']

        runner.train(train_loader, validation_loader, optimizer, criterion, num_epochs, window_size, 5)
        print("✅ Training finished.")

        print("\n" + "="*10 + " TESTING " + "="*10)
        y_pred, y_true = runner.test(test_loader, window_size, 5)
        process_test_results(y_pred, y_true)
if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="Train and evaluate DGDNN model for a specific stock market.")
    parser.add_argument('--model', 
                        type=str, 
                        required=True, 
                        choices=['dgdnn', 'graphwavenet', 'darnn', 'hyperstockgraph'],
                        help="The model to run.) #Choose from 'dgdnn', 'graphwavenet', 'darnn', or 'hyperstockgraph'.")
        # Add the required --market argument
    parser.add_argument(
        '--market',
        type=str,
        required=True,
        choices=['nasdaq', 'nyse', 'sse'],
        help="The stock market to process (e.g., 'nasdaq', 'nyse', 'sse'). This name is used to find the corresponding config and tickers file."
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function, passing the parsed arguments
    main(args)