import csv
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm


def retrieve_company_list(tickers_csv_path: Path) -> List[str]:
    """
    Reads a list of company tickers from a specified CSV file.

    Args:
        tickers_csv_path (Path): The path to the CSV file containing tickers.
                                 It's expected to have tickers in the first column.

    Returns:
        List[str]: A list of company ticker symbols.
    """
    company_list = []
    try:
        with open(tickers_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                if row:  # Ensure the row is not empty
                    company_list.append(row[0])
    except FileNotFoundError:
        print(f"❌ Error: Ticker file not found at {tickers_csv_path}")
        return []
    except Exception as e:
        print(f"An error occurred while reading the ticker file: {e}")
        return []
        
    print(f"Retrieved {len(company_list)} tickers from {tickers_csv_path.name}.")
    return company_list


def filter_stocks_from_timeperiod(
    company_list: List[str],
    market: str,
    total_time_period: List[str],
    hist_price_stocks_path: Path
) -> List[str]:
    """
    Filters a list of stocks to include only those with complete historical data
    for a given time period.

    Args:
        company_list (List[str]): The initial list of company tickers.
        market (str): The name of the market (e.g., 'NASDAQ', 'NYSE'), used for constructing file paths.
        total_time_period (List[str]): A list containing the start and end dates ['YYYY-MM-DD', 'YYYY-MM-DD'].
        hist_price_stocks_path (Path): The base directory path where historical price CSVs are stored.

    Returns:
        List[str]: A filtered list of company tickers that have sufficient data.
    """
    filtered_list = []
    start_date_str, end_date_str = total_time_period[0], total_time_period[1]
    
    print(f"\nFiltering stocks for the period: {start_date_str} to {end_date_str}...")
    
    # Use tqdm for a progress bar
    for company in tqdm(company_list, desc="Checking stock data availability"):
        # Construct the path to the individual stock's CSV file
        # This assumes a structure like .../hist_prices/NASDAQ/AAPL.csv
        market_path = hist_price_stocks_path
        stock_file_path = market_path / f"{market.upper()}_{company}.csv"
        
        if not stock_file_path.exists():
            # Silently skip if the file doesn't exist, or uncomment below to be verbose
            # print(f"Info: Skipping {company} - CSV file not found.")
            continue
            
        try:
            df = pd.read_csv(stock_file_path)
            
            # Ensure the 'Date' column exists and is not empty
            if 'Date' not in df.columns or df['Date'].isnull().all():
                continue
                
            # Convert 'Date' column to datetime objects
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Check if the stock's data range covers the required period
            first_date = df['Date'].iloc[0]
            last_date = df['Date'].iloc[-1]
            
            required_start_date = pd.to_datetime(start_date_str)
            required_end_date = pd.to_datetime(end_date_str)
            
            if first_date <= required_start_date and last_date >= required_end_date:
                filtered_list.append(company)

        except pd.errors.EmptyDataError:
            # Handle cases where the CSV file is empty
            # print(f"Info: Skipping {company} - Empty CSV file.")
            continue
        except Exception as e:
            print(f"Warning: Could not process {company}. Error: {e}")
            continue
            
    print(f"\nFiltering complete. {len(filtered_list)} out of {len(company_list)} stocks have sufficient data for the period.")
    return filtered_list