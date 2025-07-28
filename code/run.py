import argparse
import sys
from pathlib import Path
import os

# Add the models directory to Python path
PROJECT_PATH = Path(__file__).parent.resolve()
sys.path.append(str(PROJECT_PATH))
sys.path.append(str(PROJECT_PATH / "models" / "DGDNN" / "Model"))
sys.path.append(str(PROJECT_PATH / "models" / "HyperStockGAT" / "training"))
sys.path.append(str(PROJECT_PATH / "models" / "HyperStockGAT" / "training" / "layers"))
sys.path.append(str(PROJECT_PATH / "models" / "HyperStockGAT" / "training" / "utilities"))
sys.path.append(str(PROJECT_PATH / "models" / "HyperStockGAT" / "training" / "models"))
sys.path.append(str(PROJECT_PATH / "models" / "HyperStockGAT"))
sys.path.append(str(PROJECT_PATH / "model_runners"))




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
from utils import load_model, remove_unshaped_samples

torch.manual_seed(42)  # For reproducibility

# --- Local Imports ---
# Ensure these paths are correct relative to your project structure.
# You might need to adjust them if your project layout is different.
try:
    from dataset_utils import filter_stocks_from_timeperiod, retrieve_company_list
    from data.geometric_dataset_gen import MyDataset as MyGeometricDataset
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

    print(train_dataset[0])


    num_nodes =len(filtered_company_list)
    #train_dataset = remove_unshaped_samples(train_dataset, num_nodes, window_size, 5)
    n_features = 5 # Set the number of features
    print(f"Number of nodes (stocks): {num_nodes} {train_dataset[0].x.shape=}")

    # ------------------ 4. BUILD THE MODEL FROM CONFIG PARAMS ------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    
    # ------------------ 5. TRAIN AND TEST THE MODEL USING THE RUNNER ------------------s
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
        runner = DGDNNRunner(model_DGDNN, device, market_name)

        # build the optimizer & criterion
        optimizer = optim.Adam(model_DGDNN.parameters(), lr=float(train_param['learning_rate']), weight_decay=float(train_param['weight_decay']))
        criterion = nn.BCEWithLogitsLoss()
        num_epochs = train_param['epochs']
        alpha = train_param.get('neighbour_radius_coeff', 0.0)

        # Train the model
        runner.train(
            train_dataset, validation_dataset, optimizer, criterion, num_epochs,
            alpha, neighbor_distance_regularizer, theta_regularizer, window_size, num_nodes,
            batch_size=8)

        print("\n" + "="*10 + " TESTING " + "="*10)
        
        # Test the model
        y_pred, y_true = runner.test(test_dataset, window_size, num_nodes)

        # Test the results 
        process_test_results(y_pred, y_true)

        
    elif args.model == 'graphwavenet':
        print("GraphWaveNet model selected. Running training and evaluation pipeline.")
        GWN = load_model('GraphWaveNet')
        model_param = model_param['GraphWaveNet']
        model_GWN = GWN(
            device=device,              # The computing device (e.g., 'cuda' for GPU, 'cpu' for CPU) where the model and its tensors will reside.
            num_nodes=num_nodes,        # The total number of nodes in the graph.
            #dropout=0.3,               # (Optional) The dropout rate for regularization. If uncommented, a fraction of neurons will be randomly set to zero during training to prevent overfitting.
            supports=None,              # A list of static adjacency matrices or graph supports. If provided, these are used for graph convolutions. If None, the model might learn an adaptive adjacency matrix.
            gcn_bool=True,              # A boolean flag indicating whether to use Graph Convolutional Network (GCN) layers in the model.
            addaptadj=True,             # A boolean flag indicating whether to learn an adaptive adjacency matrix. If True, the model will dynamically learn graph connections.
            aptinit=None,               # (Optional) Initial values for the adaptive adjacency matrix. If 'addaptadj' is True and 'aptinit' is provided, it initializes the adaptive matrix. If None, it's typically initialized randomly.
            in_dim=n_features,          # The input feature dimension per node. This is the number of features describing each node at each time step.
            out_dim=1,                  # The output feature dimension per node. For tasks like forecasting a single value (e.g., next stock price), this would be 1.
            residual_channels=32,      # The number of channels used in the residual connections within the WaveNet architecture.
            dilation_channels=32,      # The number of channels used in the dilated convolutional layers. These layers are responsible for capturing temporal dependencies.
            skip_channels=32,          # The number of channels used in the skip connections. Skip connections help in propagating information directly to the output layers, mitigating vanishing gradients.
            end_channels=256,           # The number of channels in the final output layers of the model, typically before the final prediction head.
            kernel_size=3,              # The size of the kernel for the temporal convolutional layers. A kernel size of 1 means it operates on individual time steps.
            blocks=1,                   # The number of sequential blocks in the Graph WaveNet architecture. Each block typically contains multiple layers.
            layers=8                    # The number of dilated convolutional layers within each block. The dilation factor typically increases with each layer.
        ).to(device)


        runner = GraphWaveNetRunner(model_GWN, device, market_name)
        print(f"Model parameters: {sum([p.numel() for p in model_GWN.parameters()]):,}")

        optimizer = optim.Adam(model_GWN.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        num_epochs = train_param['epochs']

        runner.train(
            train_dataset, 
            validation_dataset, 
            optimizer, 
            criterion, 
            num_epochs, 
            window_size, 
            5,  # num_features
            batch_size=16  # Add batch_size parameter
        )

        print("\n" + "="*10 + " TESTING " + "="*10)
        y_pred, y_true = runner.test(test_dataset, window_size, 5)
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
        runner = DARNNRunner(model_DARNN, device, market_name)
        optimizer = optim.Adam(model_DARNN.parameters(), lr=float(train_param['learning_rate']), weight_decay=float(train_param['weight_decay']))
        criterion = nn.BCEWithLogitsLoss()
        runner.train(train_dataset, validation_dataset, optimizer, criterion, train_param['epochs'], seq_length=window_size)
        y_pred, y_true = runner.test(test_dataset, seq_length=window_size, num_features=5)
        process_test_results(y_pred, y_true)   


    elif args.model == 'hyperstockgraph':
        from model_runners.hyperstockgraph_runner import HyperStockGraphRunner
        NCModel = load_model('HyperStockGraph')
        
        args = argparse.Namespace(
            p='../data/2013-01-01',
            m='NASDAQ',
            t=None,
            l=window_size,
            u=256,
            s=10,
            r=1e-3,
            a=10,
            gpu=0,
            emb_file='NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy',
            rel_name='sector_industry',
            inner_prod=0,
            lr=0.001,
            dropout=0.2,
            model='HGCN',
            dim=256,
            manifold='PoincareBall',
            c=1.0,
            cuda=0,
            epochs=5000,
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
            pos_weight=0,
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
            device = device,
            num_feat = 5,  # Assuming each node has 5 features
            num_nodes = num_nodes,  # Number of nodes in the graph
            n_classes = 1
        )
   

        model_HSG = NCModel(args).to(device)

        runner = HyperStockGraphRunner(model_HSG, device, market_name)
        print(f"Model parameters: {sum([p.numel() for p in model_HSG.parameters()]):,}")
        print("Model created successfully:")

        optimizer = optim.Adam(model_HSG.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        num_epochs = train_param['epochs']

        runner.train(train_dataset, validation_dataset, optimizer, criterion, num_epochs, window_size, 5)
        print("✅ Training finished.")

        print("\n" + "="*10 + " TESTING " + "="*10)
        y_pred, y_true = runner.test(test_dataset, window_size, 5)
        process_test_results(y_pred, y_true)




if __name__ == '__main__':

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()
    writer.close()
    # Create the parser
    parser = argparse.ArgumentParser(description="Train and evaluate a model for a specific stock market.")

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