import os
import csv
import torch
from models.DGDNN.Data.geometric_dataset_gen import MyGeometricDataset

# This function replaces the manual ticker loading loop
def load_company_tickers(ticker_dir, market_name):
    """Loads a list of company tickers from a CSV file."""
    path = os.path.join(ticker_dir, f"{market_name}.csv")
    company_list = []
    with open(path) as f:
        file = csv.reader(f)
        for line in file:
            company_list.append(line[0])
    return company_list

def get_datasets(config):
    """
    Initializes and returns the train, validation, and test datasets
    based on the provided configuration.
    """
    # Unpack paths and parameters from the config dictionary
    p = config['dataset_params']
    paths = config['data_paths']

    # Use the company list from the config, or load it if set to 'all'
    if p['company_list'] == 'all':
        company_list = load_company_tickers(paths['ticker_dir'], p['market'])
        # You could add logic here to limit the number, e.g., company_list = company_list[:20]
    else:
        company_list = p['company_list']

    print(f"Using {len(company_list)} companies for market {p['market']}.")
    print(f"Companies: {company_list}")

    # Create the datasets
    print("-" * 5, "Building train dataset...", "-" * 5)
    train_dataset = MyGeometricDataset(
        stock_csv_dir=paths['stock_csv_dir'],
        dest_dir=paths['graph_dest_dir'],
        market=p['market'],
        company_list=company_list,
        start_date=p['train_sedate'][0],
        end_date=p['train_sedate'][1],
        window=p['window_size'],
        dataset_label='Train',
        use_fast_approximation=p['use_fast_approximation']
    )

    print("-" * 5, "Building validation dataset...", "-" * 5)
    validation_dataset = MyGeometricDataset(
        stock_csv_dir=paths['stock_csv_dir'],
        dest_dir=paths['graph_dest_dir'],
        market=p['market'],
        company_list=company_list,
        start_date=p['val_sedate'][0],
        end_date=p['val_sedate'][1],
        window=p['window_size'],
        dataset_label='Validation',
        use_fast_approximation=p['use_fast_approximation']
    )

    print("-" * 5, "Building test dataset...", "-" * 5)
    test_dataset = MyGeometricDataset(
        stock_csv_dir=paths['stock_csv_dir'],
        dest_dir=paths['graph_dest_dir'],
        market=p['market'],
        company_list=company_list,
        start_date=p['test_sedate'][0],
        end_date=p['test_sedate'][1],
        window=p['window_size'],
        dataset_label='Test',
        use_fast_approximation=p['use_fast_approximation']
    )

    return train_dataset, validation_dataset, test_dataset