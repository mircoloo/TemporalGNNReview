#!python3 
import argparse
import sys
from pathlib import Path
import os

from model_runners import *
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, mean_squared_error, recall_score
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

torch.manual_seed(42)  # For reproducibility

# --- Local Imports ---
from utils.dataset_utils import filter_stocks_from_timeperiod, retrieve_company_list
from data.geometric_dataset_gen import MyDataset as MyGeometricDataset
from utils.utils import load_model


PROJECT_PATH = Path(__file__).parent.resolve()



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
    CONFIG_FILE_PATH = PROJECT_PATH / "configs" / "main_config.yaml"
    MARKET_CONFIG__FILE_PATH = PROJECT_PATH / f"configs/{market_name}_config.yaml"
    MODELS_WEIGHTS_DIR_PATH = PROJECT_PATH / "models/weights"
    
    # Create weights directory if it doesn't exist
    MODELS_WEIGHTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    
    # Assuming data is in a fixed relative location
    hist_price_stocks_path = PROJECT_PATH / f"data/datasets/hist_prices/{market_name.upper()}"
    graph_dest_path = PROJECT_PATH / "data/datasets/graph"
    tickers_csv_path = PROJECT_PATH / f"data/tickers/{market_name.upper()}.csv"

    # Load market-specific configuration file
    try:
        with open(MARKET_CONFIG__FILE_PATH, 'r') as f:
            config_yaml = yaml.safe_load(f)
        with open(CONFIG_FILE_PATH, 'r') as config_file:
            main_config_yaml = yaml.safe_load(config_file)
    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found at {MARKET_CONFIG__FILE_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error parsing YAML config: {repr(e)}")
        sys.exit(1)

    # ------------------ 2. ASSIGN VARIABLES FROM DATASET CONFIG ------------------
    dataset_param = config_yaml['dataset_params']
    model_param = config_yaml['model_params']
    train_param = config_yaml['training_params']

    market = dataset_param['market']
    train_sedate = dataset_param['train_sedate']
    val_sedate = dataset_param['val_sedate']
    test_sedate = dataset_param['test_sedate']
    window_size = dataset_param['window_size']
    use_fast_approximation = dataset_param['use_fast_approximation']
    batch_size = train_param.get('batch_size', 1)  # Default to 1 if not specified
    
    # ------------------ 3. LOAD AND PREPARE DATASET ------------------
    company_list = retrieve_company_list(tickers_csv_path)
    # Retrieve the time period 
    total_time_period = [min(train_sedate + val_sedate + test_sedate), max(train_sedate + val_sedate + test_sedate)]
    
    # Filters only the company which respects the time period
    filtered_company_list = filter_stocks_from_timeperiod(company_list, market, total_time_period, hist_price_stocks_path)

    print(f"Original company list length: {len(company_list)}")
    print(f"Filtered company list length: {len(filtered_company_list)}")
    norm_method = args.norm.lower() if args.norm else ''
    use_adj_norm = False if args.adjnorm=='False' else True
    # Build or retrieve the datasets
    print("-" * 5, "Building train dataset...", "-" * 5)
    train_dataset = MyGeometricDataset(hist_price_stocks_path, graph_dest_path, market, filtered_company_list, train_sedate[0], train_sedate[1], window_size, 'Train', use_fast_approximation, normalize_method=norm_method, train_dates=train_sedate, minmax_normalize_adj=use_adj_norm)
    print("-" * 5, "Building validation dataset...", "-" * 5)
    validation_dataset = MyGeometricDataset(hist_price_stocks_path, graph_dest_path, market, filtered_company_list, val_sedate[0], val_sedate[1], window_size, 'Validation', use_fast_approximation, normalize_method=norm_method, train_dates=train_sedate, minmax_normalize_adj=use_adj_norm)
    print("-" * 5, "Building test dataset...", "-" * 5)
    test_dataset = MyGeometricDataset(hist_price_stocks_path, graph_dest_path, market, filtered_company_list, test_sedate[0], test_sedate[1], window_size, 'Test', use_fast_approximation, normalize_method=norm_method, train_dates=train_sedate, minmax_normalize_adj=use_adj_norm)

    num_nodes =len(filtered_company_list)
    features = main_config_yaml["features"]
    print(f"{features=}")
    n_features = len(features)
    print(f"Number of nodes (stocks): {num_nodes} {train_dataset[0].x.shape=}")

    # ------------------ 4. BUILD THE MODEL FROM CONFIG PARAMS ------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ------------------ 5. TRAIN AND TEST THE MODEL USING THE RUNNER ------------------s
    #model_orchestrator( args.model, model_param )
    if args.model == 'dgdnn':
        print("DGDNN model selected. Running training and evaluation pipeline.")
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
        
        runner = DGDNNRunner(model_DGDNN, device, market_name)

        # build the optimizer & criterion
        optimizer = optim.Adam(model_DGDNN.parameters(), lr=float(train_param['learning_rate']), weight_decay=float(train_param['weight_decay']))
        criterion = nn.BCEWithLogitsLoss()
        
        num_epochs = train_param['epochs']
        
        alpha = train_param.get('neighbour_radius_coeff', 0.0)

        # Train the model
        runner.train(
            train_dataset, 
            validation_dataset, 
            optimizer, 
            criterion, 
            num_epochs,
            alpha, 
            window_size, 
            num_nodes,
            batch_size=batch_size)
            
        runner.test(test_dataset, 
                    window_size, 
                    num_nodes)


        
    elif args.model == 'graphwavenet':
        print("GraphWaveNet model selected. Running training and evaluation pipeline.")
        GWN = load_model('GraphWaveNet')
        model_param = model_param['GraphWaveNet']
        
        
        # Prepare model config
        model_config = {
            'num_nodes': num_nodes,
            'in_dim': n_features,
            'out_dim': 1,
            'residual_channels': model_param['residual_channels'],
            'dilation_channels': model_param['dilation_channels'],
            'skip_channels': model_param['skip_channels'],
            'end_channels': model_param['end_channels'],
            'kernel_size': model_param['kernel_size'],
            'blocks': model_param['blocks'],
            'layers': model_param['layers'],
            'dropout': 0.3,
            'gcn_bool': True,
            'addaptadj': True,
        }
        
        model_GWN = GWN(
            device=device,
            **model_config
        ).to(device)

        runner = GraphWaveNetRunner(model_GWN, device, market_name)
        print(f"Model parameters: {sum([p.numel() for p in model_GWN.parameters()]):,}")
        print(f"Creating GraphWaveNet model with parameters: {model_param}")
        print(f"Learning rate: {train_param['learning_rate']}, weight decay: {train_param['weight_decay']}")
        print(f"Batch size: {batch_size}, Epochs: {train_param['epochs']}")
        # Training setup
        optimizer = optim.Adam(model_GWN.parameters(), lr=float(train_param['learning_rate']), weight_decay=float(train_param['weight_decay']))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([.8]).to(device))
        runner.train(train_dataset, validation_dataset, optimizer, criterion, train_param['epochs'], window_size, n_features, batch_size=batch_size, threshold=.5)
        print("✅ Training finished.")
        print("\n" + "="*10 + " TESTING " + "="*10)
        runner.test(test_dataset, window_size, n_features, batch_size=batch_size, config=model_config)
    elif args.model == 'darnn':
        MultiStockDARNN = load_model('DARNN')
        model_param = model_param['DARNN']
        print(f"Creating DARNN model with parameters: {model_param}")
        model_DARNN = MultiStockDARNN(
            N = num_nodes,
            M = model_param['M'],
            P = model_param['P'],
            T=window_size-1,
            num_stocks=num_nodes,
            device=device
        ).to(device)

        print(f"Model parameters: {sum([p.numel() for p in model_DARNN.parameters()]):,}")
        print(f"Learning rate: {train_param['learning_rate']}, weight decay: {train_param['weight_decay']}")
        print(f"Batch size: {batch_size}, Epochs: {train_param['epochs']}")
        runner = DARNNRunner(model_DARNN, device, market_name)
        optimizer = optim.Adam(model_DARNN.parameters(), lr=float(train_param['learning_rate']), weight_decay=float(train_param['weight_decay']))
        criterion = nn.BCEWithLogitsLoss()
        runner.train(train_dataset, validation_dataset, optimizer, criterion, train_param['epochs'], seq_length=window_size)
        runner.test(test_dataset)
    
    elif args.model == 'dtml':
        model_DTML = load_model('DTML')
        model_param = model_param['DTML']
        print(f"Creating DTML model with parameters: {model_param}")
        model_DTML = model_DTML(
            input_size=n_features,
            hidden_size=model_param['hidden_size'],
            num_layers=model_param['num_layers'],
            n_heads=model_param['n_heads'],
            beta=model_param['beta'],
            drop_rate=model_param['drop_rate']
        ).to(device)

        print(f"Model parameters: {sum([p.numel() for p in model_DTML.parameters()]):,}")
        print(f"Learning rate: {train_param['learning_rate']}, weight decay: {train_param['weight_decay']}")
        print(f"Batch size: {batch_size}, Epochs: {train_param['epochs']}")
        runner = DTMLRunner(model_DTML, device, market_name, dates=[train_sedate, val_sedate, test_sedate], window_size=window_size, market=market)
        optimizer = optim.Adam(model_DTML.parameters(), lr=float(train_param['learning_rate']), weight_decay=float(train_param['weight_decay']))
        criterion = nn.BCEWithLogitsLoss()
        num_epochs = train_param['epochs']
        runner.train(train_dataset, validation_dataset, optimizer, criterion, num_epochs, n_features, batch_size=batch_size)
        print("✅ Training finished.")
        print("\n" + "="*10 + " TESTING " + "="*10)
        runner.test(test_dataset, batch_size=1)

    elif args.model == 'hyperstockgat':
        NCModel = load_model('hyperstockgat')
        
        args = argparse.Namespace(
            #p='../data/2013-01-01',
            #m='NASDAQ',
            device = device,
            feat_dim = 5,  # Assuming each node has 5 features
            n_nodes = num_nodes,  # Number of nodes in the graph
            n_classes = 1,
            #t=None,
            l=window_size,
            u=256,
            s=10,
            r=1e-3,
            a=10,
            gpu=0,
            #emb_file='NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy',
            #rel_name='sector_industry',
            #inner_prod=0,
            lr=0.001,
            dropout=0.2,
            model='HGCN',
            dim=6, #input dim for the decoder(?)
            manifold='Hyperboloid',
            c=1.0,
            cuda=0,
            #epochs=5000,
            weight_decay=0.0001,
            optimizer='Adam',
            momentum=0.999,
            patience=100,
            seed=None,
            log_freq=5,
            eval_freq=1,
            save=0,
            save_dir=None,
            sweep_c=0,
            lr_reduce_freq=None,
            gamma=0.5,
            print_epoch=True,
            grad_clip=True,
            min_epochs=100,
            task='nc',
            pretrained_embeddings=None,
            num_layers=10,
            bias=1,
            act='relu',
            n_heads=2,
            alpha=0.2,
            double_precision=0,
            use_att=0,
            dataset='pubmed',
            val_prop=0.05,
            test_prop=0.1,
            use_feats=1,
            normalize_feats=1,
            normalize_adj=1,
            split_seed=1234,
            
        )
   

        model_HSG = NCModel(args).to(device)

        runner = HyperStockGATRunner(model_HSG, device, market_name) 
        print(f"Model parameters: {sum([p.numel() for p in model_HSG.parameters()]):,}")
        print("Model created successfully:")

        optimizer = optim.Adam(model_HSG.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        num_epochs = train_param['epochs']

        runner.train(train_dataset, validation_dataset, optimizer, criterion, num_epochs, window_size, 5)
        print("✅ Training finished.")

        print("\n" + "="*10 + " TESTING " + "="*10)
        y_pred, y_true = runner.test(test_dataset, window_size, 5)
   




if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description="Train and evaluate a model for a specific stock market.")

    parser.add_argument('--model', 
                        type=str, 
                        required=True, 
                        choices=['dgdnn', 'graphwavenet', 'darnn', 'hyperstockgat', 'dtml'],
                        help="The model to run.) #Choose from 'dgdnn', 'graphwavenet', 'darnn', or 'hyperstockgat'.")
    # Add the required --market argument
    parser.add_argument(
        '--market',
        type=str,
        required=True,
        choices=['nasdaq', 'nyse', 'sse'],
        help="The stock market to process (e.g., 'nasdaq', 'nyse', 'sse'). This name is used to find the corresponding config and tickers file."
    )

    parser.add_argument(
        '--norm',
        type=str,
        required=True,
        choices=['zscore', 'minmax', 'log1p', 'none'],
        help="The normalization technique to apply (e.g., 'zscore', 'minmax', 'none')."
    )

    parser.add_argument(
        '--adjnorm',
        type=str,
        required=False,
        choices=['True', 'False'],
        help="Whether to apply adjacency normalization (e.g., 'True', 'False').",
        default='False'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function, passing the parsed arguments
    main(args)