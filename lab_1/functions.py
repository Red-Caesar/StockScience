from typing import Any
import pandas as pd
import numpy as np


def get_df_with_return(
    df: pd.DataFrame,
    stocks_id: str="SECID",
    price_name: str="LEGALCLOSEPRICE",
    trade_date: str="TRADEDATE",
    first_day_value: Any=None,
    ) -> pd.DataFrame:

    tickers = df[stocks_id].unique()
    df = df.sort_values(by=trade_date)
    df["RETURN"] = 0.0
    
    for ticker in tickers:
        ticker_window = (df.SECID == ticker)
        cur_price = df[ticker_window][price_name]
        prev_price = df[ticker_window][price_name].shift(1)
        df.loc[ticker_window, "RETURN"] = np.log(cur_price / prev_price)

    if first_day_value is not None:
        df["RETURN"] = df["RETURN"].fillna(first_day_value)
    return df
