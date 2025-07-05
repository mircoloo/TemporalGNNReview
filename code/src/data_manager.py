import torch
import pandas as pd

import tqdm
from pathlib import Path
import os 
import yfinance as yf
import pytz
from datetime import datetime

def create_csv_from_ticker(ticker: str, dest_folder_path: str,filename: str,start_date = None, end_date = None ) -> bool:
    df = yf.download(ticker, start=start_date, end=end_date)  # Invalid ticker
    if df.empty:
        return False
    df.columns = df.columns.get_level_values(0)
    df.to_csv(f"{dest_folder_path}/{filename}.csv", sep=',', index=True)
    return True

def read_tickers_from_csv(filepath: str) -> str:
    with open(filepath, 'r') as f:
        return f.read()
    

def create_tickers_file_from_csvs(name: str, filepath: str):
    csvs_path: Path =  Path.cwd() / 'code/data/datasets/hist_prices/America_Stocks_Updated'
    files: list[str] = os.listdir(csvs_path)
    tickers: list[str] = []
    for filename in files:
        
        splitted = filename.split('_') 
        if splitted[0] == 'NASDAQ':
            ticker = splitted[1]
            tickers.append(ticker)

    filepath: Path = Path(filepath) / Path(name)

    with open(filepath, 'w+') as f:
        for ticker in sorted(tickers):
            f.write(f"{ticker}\n")



def main():
    create_tickers_file_from_csvs("NASDAQ_new.csv", '/home/mbisoffi/tests/TemporalGNNReview/code/data/tickers')
    

if __name__ == '__main__':
    main() 
        