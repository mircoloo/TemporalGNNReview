import argparse
import sys
from pathlib import Path
import os
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime
from copy import deepcopy
import traceback

# Add project path to sys.path
PROJECT_PATH = Path(__file__).parent.resolve()
sys.path.append(str(PROJECT_PATH))

from model_runners import *
from utils.dataset_utils import filter_stocks_from_timeperiod, retrieve_company_list
from data.geometric_dataset_gen import MyDataset as MyGeometricDataset
from utils.utils import load_model

class RandomSearch:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.project_path = PROJECT_PATH
        self.results = []
        self.best_val_score = float('inf') # Initialize to infinity for minimization (Loss)
        self.best_params = None
        self.best_model_state = None
        self.best_metrics = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load base configs
        self.load_configs()
        self.prepare_data()
        
    def load_configs(self):
        market_name = self.args.market.lower()
        config_file_path = self.project_path / "configs" / "main_config.yaml"
        market_config_path = self.project_path / f"configs/{market_name}_config.yaml"
        
        try:
            with open(market_config_path, 'r') as f:
                self.config_yaml = yaml.safe_load(f)
            with open(config_file_path, 'r') as f:
                self.main_config_yaml = yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Error loading configs: {e}")
            sys.exit(1)
            
        self.dataset_param = self.config_yaml['dataset_params']
        self.model_param = self.config_yaml['model_params']
        self.train_param = self.config_yaml['training_params']
        
    def prepare_data(self):
        market_name = self.args.market.lower()
        hist_price_stocks_path = self.project_path / f"data/datasets/hist_prices/{market_name.upper()}"
        graph_dest_path = self.project_path / "data/datasets/graph"
        tickers_csv_path = self.project_path / f"data/tickers/{market_name.upper()}.csv"
        
        train_sedate = self.dataset_param['train_sedate']
        val_sedate = self.dataset_param['val_sedate']
        test_sedate = self.dataset_param['test_sedate']
        self.window_size = self.dataset_param['window_size']
        use_fast_approximation = self.dataset_param['use_fast_approximation']
        
        company_list = retrieve_company_list(tickers_csv_path)
        total_time_period = [min(train_sedate + val_sedate + test_sedate), max(train_sedate + val_sedate + test_sedate)]
        filtered_company_list = filter_stocks_from_timeperiod(company_list, self.dataset_param['market'], total_time_period, hist_price_stocks_path)
        
        self.num_nodes = len(filtered_company_list)
        self.n_features = len(self.main_config_yaml["features"])
        
        norm_method = self.args.norm.lower() if self.args.norm else ''
        use_adj_norm = False if self.args.adjnorm=='False' else True
        adjnorm_threshold = self.args.adjnorm_threshold if use_adj_norm else 0.0
        
        print("-" * 5, "Building datasets...", "-" * 5)
        self.train_dataset = MyGeometricDataset(hist_price_stocks_path, graph_dest_path, self.dataset_param['market'], filtered_company_list, train_sedate[0], train_sedate[1], self.window_size, 'Train', use_fast_approximation, normalize_method=norm_method, train_dates=train_sedate, minmax_normalize_adj=use_adj_norm, threshold=adjnorm_threshold)
        self.validation_dataset = MyGeometricDataset(hist_price_stocks_path, graph_dest_path, self.dataset_param['market'], filtered_company_list, val_sedate[0], val_sedate[1], self.window_size, 'Validation', use_fast_approximation, normalize_method=norm_method, train_dates=train_sedate, minmax_normalize_adj=use_adj_norm, threshold=adjnorm_threshold)
        self.test_dataset = MyGeometricDataset(hist_price_stocks_path, graph_dest_path, self.dataset_param['market'], filtered_company_list, test_sedate[0], test_sedate[1], self.window_size, 'Test', use_fast_approximation, normalize_method=norm_method, train_dates=train_sedate, minmax_normalize_adj=use_adj_norm, threshold=adjnorm_threshold)

    def get_search_space(self, model_name):
        # Batch size is fixed to 1 as requested
        common_space = {
            'learning_rate': [1e-3, 5e-4, 2e-4, 1e-4, 5e-5],
            'weight_decay': [1e-4, 1e-5, 1.5e-5, 1e-6],
        }
        
        model_spaces = {
            'dgdnn': {
                'neighbour_radius_coeff': [1e-3, 2.7e-3, 2.9e-3, 5e-3, 8.6e-3],
                'embedding_hidden_size': [128, 256, 512],
                'embedding_output_size': [64, 128, 256],
                'num_heads': [2, 3, 4],
                'layers': [2, 3, 4, 5, 6, 7, 8],
                'expansion_step': [3, 5, 7, 9],
            },
            'graphwavenet': {
                'dropout': [0.1, 0.3, 0.5],
                'residual_channels': [32, 64, 128],
                'dilation_channels': [32, 64, 128],
                'skip_channels': [32, 64, 128],
                'blocks': [2, 3, 4],
                'layers': [2, 3],
                'kernel_size': [2, 3, 4],
            },
            'darnn': {
                'M': [16,32, 64, 128, 256],
                'P': [16, 32, 64, 128, 256],
            },
            'dtml': {
                'hidden_size': [32, 64, 128, 256],
                'n_heads': [2, 4, 8],
                'beta': [0.1, 0.5, 1.0, 2.0],
                'drop_rate': [0.1, 0.2, 0.3],
                'num_layers': [2, 3, 4]
            },
            'hyperstockgat': {
                'dim': [16, 32, 64, 128, 256],
                'dropout': [0.1, 0.3, 0.5],
                'alpha': [0.1, 0.2],
                'num_layers': [2, 3, 4, 5],
            }
        }
        
        space = common_space.copy()
        if model_name in model_spaces:
            space.update(model_spaces[model_name])
        return space

    def sample_params(self, model_name):
        space = self.get_search_space(model_name)
        params = {}
        for k, v in space.items():
            params[k] = random.choice(v)
        
        # Enforce batch_size = 1
        params['batch_size'] = 1
        return params

    def run(self):
        print(f"🚀 Starting Random Search for {self.args.model.upper()} on {self.args.market.upper()}")
        print(f"Running {self.args.n_trials} trials...")
        
        for trial in range(self.args.n_trials):
            print(f"\n{'='*20} Trial {trial+1}/{self.args.n_trials} {'='*20}")
            
            # Sample parameters
            params = self.sample_params(self.args.model)
            print(f"Parameters: {params}")
            
            try:
                val_results, model_state = self.train_and_evaluate(params)
                
                # Use Validation Loss as the primary metric for optimization (Minimization)
                val_score = val_results.get('loss', float('inf'))
                
                result = {
                    'trial': trial + 1,
                    'params': params,
                    'val_score': val_score,
                    'metrics': val_results,
                    'status': 'success'
                }
                self.results.append(result)
                
                print(f"Trial {trial+1} Result: Val Score (Loss) = {val_score:.4f}")
                print(f"Metrics: {val_results}")
                
                if val_score < self.best_val_score:
                    print(f"🌟 New Best Score! ({self.best_val_score:.4f} -> {val_score:.4f})")
                    self.best_val_score = val_score
                    self.best_params = params
                    self.best_model_state = model_state
                    self.best_metrics = val_results
                
                # Save results incrementally after each successful trial
                self.save_results()
                    
            except Exception as e:
                print(f"❌ Trial {trial+1} Failed: {e}")
                traceback.print_exc()
                self.results.append({
                    'trial': trial + 1,
                    'params': params,
                    'status': 'failed',
                    'error': str(e)
                })
                # Save results incrementally even after failure
                self.save_results()
        
        self.finish()

    def train_and_evaluate(self, params):
        model_name = self.args.model
        batch_size = params['batch_size']
        lr = params['learning_rate']
        wd = params['weight_decay']
        
        # Initialize model based on type
        if model_name == 'dgdnn':
            DGDNN = load_model('DGDNN')
            base_params = self.model_param['DGDNN']
            
            # Dynamic parameter generation for DGDNN
            layers = params.get('layers', base_params['layers'])
            embedding_output_size = params.get('embedding_output_size', base_params['embedding_output_size'])
            raw_feature_size = base_params['raw_feature_size']
            
            # Generate lists if layers changed or if we are in random search
            # Diffusion size: [5*window, emb_out, emb_out, ...] (Simplified)
            # Or better: keep the structure of increasing/decreasing if possible, but for random search flat is safer.
            # Let's use a flat structure for simplicity in random search: [input] + [128] * layers
            diffusion_size = [5 * self.window_size] + [128] * layers
            
            # Embedding size calculation
            # Layer 0: diffusion_size[1] + raw_feature_size
            # Layer > 0: diffusion_size[i+1] + embedding_output_size
            embedding_size = []
            for i in range(layers):
                if i == 0:
                    emb_dim = diffusion_size[1] + raw_feature_size
                else:
                    emb_dim = diffusion_size[i+1] + embedding_output_size
                embedding_size.append(emb_dim)
                
            active_layers = [True] * layers

            model = DGDNN(
                diffusion_size=diffusion_size,
                embedding_size=embedding_size,
                embedding_hidden_size=params.get('embedding_hidden_size', base_params['embedding_hidden_size']),
                embedding_output_size=embedding_output_size,
                raw_feature_size=raw_feature_size,
                classes=1,
                layers=layers,
                num_nodes=self.num_nodes,
                expansion_step=params.get('expansion_step', base_params['expansion_step']),
                num_heads=params.get('num_heads', base_params['num_heads']),
                active=active_layers
            ).to(self.device)
            runner = DGDNNRunner(model, self.device, self.args.market)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            criterion = nn.BCEWithLogitsLoss()
            
            runner.train(
                self.train_dataset, 
                self.validation_dataset, 
                optimizer, 
                criterion, 
                self.args.epochs,
                params.get('neighbour_radius_coeff', 0.0), 
                self.window_size, 
                self.num_nodes,
                batch_size=batch_size
            )
            
        elif model_name == 'graphwavenet':
            GWN = load_model('GraphWaveNet')
            base_params = self.model_param['GraphWaveNet']
            model_config = {
                'num_nodes': self.num_nodes,
                'in_dim': self.n_features,
                'out_dim': 1,
                'residual_channels': params.get('residual_channels', base_params['residual_channels']),
                'dilation_channels': params.get('dilation_channels', base_params['dilation_channels']),
                'skip_channels': params.get('skip_channels', base_params['skip_channels']),
                'end_channels': base_params['end_channels'],
                'kernel_size': params.get('kernel_size', base_params['kernel_size']),
                'blocks': params.get('blocks', base_params['blocks']),
                'layers': params.get('layers', base_params['layers']),
                'dropout': params.get('dropout', base_params['dropout']),
                'gcn_bool': base_params['gcn_bool'],
                'addaptadj': base_params['addaptadj'],
            }
            model = GWN(device=self.device, **model_config).to(self.device)
            runner = GraphWaveNetRunner(model, self.device, self.args.market)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([.8]).to(self.device))
            
            runner.train(self.train_dataset, self.validation_dataset, optimizer, criterion, self.args.epochs, self.window_size, self.n_features, batch_size=batch_size, threshold=.5)

        elif model_name == 'darnn':
            MultiStockDARNN = load_model('DARNN')
            base_params = self.model_param['DARNN']
            model = MultiStockDARNN(
                N=self.num_nodes,
                M=params.get('M', base_params['M']),
                P=params.get('P', base_params['P']),
                T=self.window_size-1,
                num_stocks=self.num_nodes,
                device=self.device
            ).to(self.device)
            runner = DARNNRunner(model, self.device, self.args.market)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            criterion = nn.BCEWithLogitsLoss()
            runner.train(self.train_dataset, self.validation_dataset, optimizer, criterion, self.args.epochs, seq_length=self.window_size)

        elif model_name == 'dtml':
            DTML = load_model('DTML')
            base_params = self.model_param['DTML']
            model = DTML(
                input_size=self.n_features,
                hidden_size=params.get('hidden_size', base_params['hidden_size']),
                num_layers=params.get('num_layers', base_params['num_layers']),
                n_heads=params.get('n_heads', base_params['n_heads']),
                beta=params.get('beta', base_params['beta']),
                drop_rate=params.get('drop_rate', base_params['drop_rate'])
            ).to(self.device)
            runner = DTMLRunner(model, self.device, self.args.market, feature_normalization=self.args.norm, dates=[self.dataset_param['train_sedate'], self.dataset_param['val_sedate'], self.dataset_param['test_sedate']], window_size=self.window_size, market=self.dataset_param['market'])
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            criterion = nn.BCEWithLogitsLoss()
            runner.train(self.train_dataset, self.validation_dataset, optimizer, criterion, self.args.epochs, self.n_features, batch_size=batch_size)

        elif model_name == 'hyperstockgat':
            NCModel = load_model('hyperstockgat')
            base_params = self.model_param.get('HyperStockGAT', {})
            
            # Construct args namespace for HyperStockGAT
            hsg_args = argparse.Namespace(
                device=self.device,
                feat_dim=5,
                n_nodes=self.num_nodes,
                n_classes=1,
                dim=params.get('dim', 32),
                num_layers=params.get('num_layers', 2),
                l=self.window_size,
                model='HGCN',
                manifold='PoincareBall',
                act='relu',
                dropout=params.get('dropout', 0.1),
                c=1,
                cuda=1 if torch.cuda.is_available() else -1,
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
                bias=1,
                n_heads=2,
                alpha=params.get('alpha', 0.2),
                double_precision=0,
                use_att=1,
                val_prop=0.05,
                test_prop=0.1,
                use_feats=1,
                normalize_feats=1,
                normalize_adj=1,
                split_seed=1234,
                pos_weight=0,
            )
            
            model = NCModel(hsg_args).to(self.device)
            runner = HyperStockGATRunner(model, self.device, self.args.market)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()
            runner.train(self.train_dataset, self.validation_dataset, optimizer, criterion, self.args.epochs, self.window_size, 5, batch_size=batch_size)

        # Evaluate on validation set
        # We use the runner's test method but pass the validation dataset
        # Note: Some runners might expect specific args for test, we try to handle common ones
        print("Evaluating on Validation Set...")
        if model_name == 'dgdnn':
            val_results = runner.test(self.validation_dataset, self.window_size, self.num_nodes, save_attention=False)
        elif model_name == 'graphwavenet':
            val_results = runner.test(self.validation_dataset, self.window_size, self.n_features, batch_size=batch_size, config=model_config)
        elif model_name == 'darnn':
            val_results = runner.test(self.validation_dataset)
        elif model_name == 'dtml':
            val_results = runner.test(self.validation_dataset)
        elif model_name == 'hyperstockgat':
            val_results = runner.test(self.validation_dataset)
            
        # Return all validation results (metrics) and model state
        return val_results, deepcopy(model.state_dict())

    def finish(self):
        print("\n" + "="*40)
        print("🏁 Random Search Completed")
        print("="*40)
        
        if self.best_params:
            print(f"Best Parameters found: {self.best_params}")
            print(f"Best Validation Score: {self.best_val_score:.4f}")
            
            # Final save
            self.save_results()
            
            # Run final test with best model
            print("\nRunning Final Test with Best Model...")
            self.run_final_test()
        else:
            print("No successful trials.")

    def run_final_test(self):
        # Re-instantiate model with best params and load state
        params = self.best_params
        model_name = self.args.model
        batch_size = params['batch_size']
        
        # ... (Re-instantiation logic similar to train_and_evaluate but loading state_dict) ...
        # For brevity, I'll just re-use the logic but load the state dict
        # In a real scenario, refactoring the instantiation into a method 'create_model(params)' is better.
        
        # Quick hack: Re-run train_and_evaluate but with 0 epochs? No, runners might not support 0 epochs nicely.
        # Better: Re-instantiate and load state_dict.
        
        # Re-instantiation (Simplified copy of logic)
        if model_name == 'dgdnn':
            DGDNN = load_model('DGDNN')
            base_params = self.model_param['DGDNN']
            model = DGDNN(
                diffusion_size=base_params['diffusion_size'],
                embedding_size=base_params['embedding_size'],
                embedding_hidden_size=base_params['embedding_hidden_size'],
                embedding_output_size=base_params['embedding_output_size'],
                raw_feature_size=base_params['raw_feature_size'],
                classes=1,
                layers=base_params['layers'],
                num_nodes=self.num_nodes,
                expansion_step=base_params['expansion_step'],
                num_heads=base_params['num_heads'],
                active=base_params['active_layers']
            ).to(self.device)
            runner = DGDNNRunner(model, self.device, self.args.market)
            
        elif model_name == 'graphwavenet':
            GWN = load_model('GraphWaveNet')
            base_params = self.model_param['GraphWaveNet']
            model_config = {
                'num_nodes': self.num_nodes,
                'in_dim': self.n_features,
                'out_dim': 1,
                'residual_channels': params.get('residual_channels', base_params['residual_channels']),
                'dilation_channels': params.get('dilation_channels', base_params['dilation_channels']),
                'skip_channels': params.get('skip_channels', base_params['skip_channels']),
                'end_channels': base_params['end_channels'],
                'kernel_size': base_params['kernel_size'],
                'blocks': base_params['blocks'],
                'layers': base_params['layers'],
                'dropout': params.get('dropout', base_params['dropout']),
                'gcn_bool': base_params['gcn_bool'],
                'addaptadj': base_params['addaptadj'],
            }
            model = GWN(device=self.device, **model_config).to(self.device)
            runner = GraphWaveNetRunner(model, self.device, self.args.market)

        elif model_name == 'darnn':
            MultiStockDARNN = load_model('DARNN')
            base_params = self.model_param['DARNN']
            model = MultiStockDARNN(
                N=self.num_nodes,
                M=params.get('M', base_params['M']),
                P=params.get('P', base_params['P']),
                T=self.window_size-1,
                num_stocks=self.num_nodes,
                device=self.device
            ).to(self.device)
            runner = DARNNRunner(model, self.device, self.args.market)

        elif model_name == 'dtml':
            DTML = load_model('DTML')
            base_params = self.model_param['DTML']
            model = DTML(
                input_size=self.n_features,
                hidden_size=params.get('hidden_size', base_params['hidden_size']),
                num_layers=base_params['num_layers'],
                n_heads=params.get('n_heads', base_params['n_heads']),
                beta=params.get('beta', base_params['beta']),
                drop_rate=params.get('drop_rate', base_params['drop_rate'])
            ).to(self.device)
            runner = DTMLRunner(model, self.device, self.args.market, feature_normalization=self.args.norm, dates=[self.dataset_param['train_sedate'], self.dataset_param['val_sedate'], self.dataset_param['test_sedate']], window_size=self.window_size, market=self.dataset_param['market'])

        elif model_name == 'hyperstockgat':
            NCModel = load_model('hyperstockgat')
            hsg_args = argparse.Namespace(
                device=self.device,
                feat_dim=5,
                n_nodes=self.num_nodes,
                n_classes=1,
                dim=params.get('dim', 32),
                num_layers=2,
                l=self.window_size,
                model='HGCN',
                manifold='PoincareBall',
                act='relu',
                dropout=params.get('dropout', 0.1),
                c=1,
                cuda=1 if torch.cuda.is_available() else -1,
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
                bias=1,
                n_heads=2,
                alpha=params.get('alpha', 0.2),
                double_precision=0,
                use_att=1,
                val_prop=0.05,
                test_prop=0.1,
                use_feats=1,
                normalize_feats=1,
                normalize_adj=1,
                split_seed=1234,
                pos_weight=0,
            )
            model = NCModel(hsg_args).to(self.device)
            runner = HyperStockGATRunner(model, self.device, self.args.market)

        # Load best state
        model.load_state_dict(self.best_model_state)
        
        # Run Test
        if model_name == 'dgdnn':
            test_results = runner.test(self.test_dataset, self.window_size, self.num_nodes, save_attention=True)
        elif model_name == 'graphwavenet':
            # Need model_config for GWN test
            base_params = self.model_param['GraphWaveNet']
            model_config = {
                'num_nodes': self.num_nodes,
                'in_dim': self.n_features,
                'out_dim': 1,
                'residual_channels': params.get('residual_channels', base_params['residual_channels']),
                'dilation_channels': params.get('dilation_channels', base_params['dilation_channels']),
                'skip_channels': params.get('skip_channels', base_params['skip_channels']),
                'end_channels': base_params['end_channels'],
                'kernel_size': base_params['kernel_size'],
                'blocks': base_params['blocks'],
                'layers': base_params['layers'],
                'dropout': params.get('dropout', base_params['dropout']),
                'gcn_bool': base_params['gcn_bool'],
                'addaptadj': base_params['addaptadj'],
            }
            test_results = runner.test(self.test_dataset, self.window_size, self.n_features, batch_size=batch_size, config=model_config)
        elif model_name == 'darnn':
            test_results = runner.test(self.test_dataset)
        elif model_name == 'dtml':
            test_results = runner.test(self.test_dataset)
        elif model_name == 'hyperstockgat':
            test_results = runner.test(self.test_dataset, save_attention=True)
            
        print(f"Final Test Results: {test_results}")
    
    def save_results(self):
        """Save current results to JSON file"""
        save_path = self.project_path / "analysis_results" / f"random_search_{self.args.model}_{self.args.market}_{self.timestamp}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert any non-serializable objects if necessary (though params and scores should be fine)
        # Deepcopy to avoid modifying the original list if we need to sanitize
        
        with open(save_path, 'w') as f:
            json.dump({
                'model': self.args.model,
                'market': self.args.market,
                'best_params': self.best_params,
                'best_val_score': self.best_val_score,
                'best_metrics': self.best_metrics,
                'all_results': self.results
            }, f, indent=4)
        print(f"Results saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Random Search Hyperparameter Optimization")
    parser.add_argument('--model', type=str, required=True, choices=['dgdnn', 'graphwavenet', 'darnn', 'hyperstockgat', 'dtml'])
    parser.add_argument('--market', type=str, required=True, choices=['nasdaq', 'nyse', 'sse'])
    parser.add_argument('--norm', type=str, default='zscore', choices=['zscore', 'minmax', 'log1p', 'none'])
    parser.add_argument('--adjnorm', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--adjnorm_threshold', type=float, default=0.0)
    parser.add_argument('--n_trials', type=int, default=10, help="Number of random trials")
    parser.add_argument('--epochs', type=int, default=300, help="Epochs per trial")
    
    args = parser.parse_args()
    
    searcher = RandomSearch(args)
    searcher.run()
