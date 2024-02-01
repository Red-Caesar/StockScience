import pandas as pd
import numpy as np


def get_df_with_return(df: pd.DataFrame, price_name: str="LEGALCLOSEPRICE") -> pd.DataFrame:
    tickers = df.SECID.unique()
    df["RETURN"] = 0.0
    df = df.sort_values(by="TRADEDATE")
    for ticker in tickers:
        ticker_window = (df.SECID == ticker)
        cur_price = df[ticker_window][price_name]
        prev_price = df[ticker_window][price_name].shift(1)
        df.loc[ticker_window,"RETURN" ] = np.log(cur_price / prev_price)
    return df
