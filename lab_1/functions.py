from typing import Any, List, Union, Dict
import pandas as pd
import numpy as np
from functools import reduce


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


def get_portfolio_expected_return(
    df: pd.DataFrame,
    ratios: Union[List[float], Dict[str, float]],
    stocks_id_col: str="SECID",
    return_col: str="RETURN",
) -> float:

    df_with_mean = df.groupby(by=stocks_id_col).aggregate({return_col: "mean"}).rename(columns={return_col: "MEAN"})
    if len(df_with_mean) != len(ratios):
        raise ValueError("Кол-во активов и кол-во долей не совпадает")

    expected_return = 0
    if isinstance(ratios, list):
        if not np.isclose(np.sum(ratios), 1.0):
            raise ValueError("Сумма долей не равна 1")

        stocks_mean = df_with_mean.MEAN.to_list()
        expected_return = sum(map(lambda a,b: a*b, ratios, stocks_mean))
    elif isinstance(ratios, dict):
        mean_dict = df_with_mean.to_dict()["MEAN"]
        for key, val in ratios.items():
            expected_return += val * mean_dict[key]
    else:
        raise ValueError(f"Неподходящий тип ratios: {type(ratios)}")

    return expected_return

def get_portfolio_std(
    df: pd.DataFrame,
    ratios: Union[List[float], Dict[str, float]],
    stocks_id_col: str="SECID",
    return_col: str="RETURN",
    date_col: str="TRADEDATE",
) -> float:

    new_df = dict()
    for stock in df[stocks_id_col].unique():
        stock_returns = df[df[stocks_id_col] == stock].sort_values(by=date_col)[return_col].to_list()
        new_df[stock] = stock_returns
    new_df = pd.DataFrame(new_df)
    cov_matrix = new_df.cov()

    portfolio_var = 0
    if isinstance(ratios, list):
        portfolio_var = np.matmul(np.matmul(ratios, cov_matrix), ratios)

    return np.sqrt(portfolio_var)


def get_VaR(returns: pd.Series, gamma: float=0.9) -> float:
    return (-returns).quantile(gamma)


def get_df_with_mean_std_return(df: pd.DataFrame,
    stocks_id: str="SECID",
    stock_return: str="RETURN",
    ) -> pd.DataFrame:
    
    tickers = df[stocks_id].unique()
    df["MEAN_RETURN"] = 0.0
    df["STD_RETURN"] = 0.0

    for ticker in tickers:
        ticker_window = (df.SECID == ticker)
        df.loc[ticker_window, "MEAN_RETURN"] = df.loc[ticker_window, stock_return].mean()
        df.loc[ticker_window, "STD_RETURN"] = df.loc[ticker_window, stock_return].std()
    
    return df
