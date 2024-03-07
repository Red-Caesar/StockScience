from typing import Any, List, Union, Dict
import pandas as pd
import numpy as np
from scipy import stats
import cvxpy as cp
from scipy.optimize import minimize, Bounds
import seaborn as sns
import matplotlib.pyplot as plt


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
    new_df = new_df.sort_index(axis=1)
    cov_matrix = new_df.cov()

    portfolio_var = 0
    if isinstance(ratios, list):
        portfolio_var = np.matmul(np.matmul(ratios, cov_matrix), ratios)
    elif isinstance(ratios, dict):
        sorted_ratio = sorted(ratios.items(), key=lambda x: x[0])
        sorted_ratio = [x[1] for x in sorted_ratio]
        portfolio_var = np.matmul(np.matmul(sorted_ratio, cov_matrix), sorted_ratio)

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

def get_stock_name(
    df: pd.DataFrame,
    stock_id: str,
    stocks_id_col: str="SECID",
    stocks_name_col: str="SHORTNAME"
) -> str:
    stock_name = df[df[stocks_id_col] == stock_id][stocks_name_col].unique()[0]
    return stock_name


def inversion_test(
    df: pd.DataFrame,
    companies: Union[List[str], pd.Series],
    target_col: str="RETURN",
    stocks_id_col: str="SECID",
    alpha: float=0.05
) -> pd.DataFrame:

    result_dict = {stocks_id_col: [], "hypothesis": []}
    for companie in companies:
        inv_count = 0
        data = df[df[stocks_id_col] == companie][target_col].tolist()
        n = len(data)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if data[i] > data[j]:
                    inv_count += 1
        e = (n * (n - 1)) / 4
        t_alpha = -stats.norm.ppf(alpha/2) * n**(3/2)/6
        result_dict[stocks_id_col].append(companie)
        if abs(inv_count - e) >= t_alpha:
            result_dict["hypothesis"].append("rejected")
        else:
            result_dict["hypothesis"].append("accepted")
    return pd.DataFrame(result_dict)


def autocorrelation_test(
    returns: pd.Series,
    alpha: float=0.05,
    shift: int=1,
) -> dict:
    N = len(returns)
    shift_returns = np.roll(returns, shift)
    numerator = N * np.sum(list(map(lambda x,y: x*y, returns, shift_returns))) - np.power(sum(returns), 2)
    denominator = N * np.sum(list(map(lambda x: x*x, returns))) - np.power(np.sum(returns), 2)
    coef_cor = numerator / denominator

    e = - 1 / (N-1)
    d = N * (N - 3) / ((N + 1) * np.power(N - 1, 2))

    cr = abs(coef_cor - e) / np.sqrt(d)
    q = stats.norm.ppf(1 - alpha/2)
    isTrend = cr > q
    return {'coef_criteria': cr, "trend": isTrend}


def generate_short_ratios(n: int) -> List:

    random_numbers = np.random.rand(n)
    diff = [1 - abs(q) for q in random_numbers]

    total_diff = sum(diff)
    needed = 1.0 - sum(random_numbers)

    adjust = [q * needed / total_diff for q in diff]
    new = [random_numbers[i] + adjust[i] for i in range(len(random_numbers))]
    return new


def solve_risk_aversion(
    mu: List[float],
    sigma: List[List[float]],
    aversion_value: int=1,
    with_short: bool=False
    ) -> Dict[str, float]:
    n = len(mu)
    w = cp.Variable(n)
    risk_aversion = cp.Parameter(nonneg=True)
    risk_aversion.value = aversion_value

    ret = mu.T @ w
    risk = cp.quad_form(w, sigma)

    objective = cp.Maximize(ret - risk_aversion * risk)
    if with_short:
        constraints = [cp.sum(w) == 1]
    else:
        constraints = [cp.sum(w) == 1, w >= 0.01]

    prob = cp.Problem(objective, constraints)

    prob.solve()
    solution = {
        "weights": w.value,
        "risk": np.sqrt(risk.value),
        "expected_return": ret.value,
        "risk_aversion": aversion_value
    }
    return solution


def get_random_portfolios(
    df: pd.DataFrame,
    with_short: bool=False,
    random_samples: int=10
) -> pd.DataFrame:
    n = df.SECID.nunique()

    random_portfolio_dict = {"mean": [], "std": [], "type": []}
    for _ in range(random_samples):
        if with_short:
            ratios = generate_short_ratios(n)
        else:
            ratios = np.random.dirichlet(np.ones(n)).tolist()

        weight = pd.DataFrame({"weights":ratios}, index=list(df.SECID.unique()))
        ex_return = get_portfolio_expected_return(df, weight.to_dict()["weights"])
        ex_std = get_portfolio_std(df, ratios)

        random_portfolio_dict["mean"].append(ex_return)
        random_portfolio_dict["std"].append(ex_std)
        if with_short:
            random_portfolio_dict["type"].append("with_short_sales")
        else:
            random_portfolio_dict["type"].append("not_short_sales")

    return pd.DataFrame(random_portfolio_dict)


def ret_risk(w, exp_return, cov):
    return -((w.T@exp_return) / (w.T@cov@w)**0.5)


def solve_markowitz(data: pd.DataFrame, with_short: bool=False) -> pd.DataFrame:
    w = np.ones((data.values.T.shape[0],1))*(1.0/data.values.T.shape[0])
    m = np.mean(data.values.T, axis=1)
    demeaned = data.values.T - m[:,None]
    m = m.reshape(m.shape[0],1)
    exp_return = m*w
    cov = np.cov(demeaned)
    if with_short:
        opt_bounds = Bounds(-1, 1)
    else:
        opt_bounds = Bounds(0, 1)
    opt_constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})
    res = minimize(
        ret_risk,
        w,
        args=(exp_return, cov),
        method='SLSQP',
        bounds=opt_bounds,
        constraints=opt_constraints)
    result_df = pd.DataFrame({"weights": res.x}, index=data.columns)
    return result_df


def minimize_portfolio(objective, mean_returns, covar_matrix, bounds, target_return=None):
    X = np.ones(mean_returns.shape[0])
    X = X / X.sum()
    bounds = bounds * mean_returns.shape[0]
    
    constraints = []
    constraints.append({'type':'eq', 'fun': lambda X: np.sum(X) - 1.0})
    if target_return:
        constraints.append({'type':'eq', 'args' : (mean_returns, ), 'fun': lambda X, mean_returns: target_return - np.dot(X, mean_returns)})
        
    return minimize(objective, X, args=(covar_matrix), method='SLSQP', constraints=constraints, bounds=bounds).x


def get_portfolio_risk(X, covar_matrix):
    return np.sqrt(np.dot(np.dot(X.T, covar_matrix), X))


def calculate_efficient_frontier(chosen_stock_returns, chosen_stock_characteristics, short_sales = True):
    bounds = ()
    if short_sales:
      bounds = ((-1.0, 1.0), ) # Short sales are allowed
    else:
      bounds = ((0.0, 1.0), ) # Short sales are forbidden

    cov_matrix = chosen_stock_returns.cov()
    
    min_risk_portfolio = minimize_portfolio(get_portfolio_risk,
                                        chosen_stock_characteristics.MEAN_RETURN,
                                        cov_matrix,
                                        bounds
                                        )
    
    min_risk_portfolio_std = get_portfolio_risk(min_risk_portfolio, cov_matrix)
    min_risk_portfolio_return = np.dot(min_risk_portfolio, chosen_stock_characteristics.MEAN_RETURN)
    
    
    target_range = np.linspace(min_risk_portfolio_return, chosen_stock_characteristics.STD_RETURN.max() + 0.003, 300)
    
    portfolio_stds = []
    portfolio_returns = []
    for portfolio_return in target_range:
        x = minimize_portfolio(get_portfolio_risk,
                                chosen_stock_characteristics.MEAN_RETURN,
                                cov_matrix,
                                bounds,
                                target_return=portfolio_return)
        portfolio_returns.append(np.dot(x, chosen_stock_characteristics.MEAN_RETURN))
        portfolio_stds.append(get_portfolio_risk(x, cov_matrix))
        
    return min_risk_portfolio_std, min_risk_portfolio_return, portfolio_stds, portfolio_returns


def plot_efficient_frontier(chosen_stock_characteristics, 
                            min_risk_portfolio_std, 
                            min_risk_portfolio_return, 
                            portfolio_stds, 
                            portfolio_returns, 
                            short_sales, 
                            color='red', 
                            num = 10):

    if short_sales:
        mes_short_sales = 'allowed short sales'
    else:
        mes_short_sales = 'forbidden short sales'
    
    sns.scatterplot(x=chosen_stock_characteristics.STD_RETURN,
        y=chosen_stock_characteristics.MEAN_RETURN, color='dodgerblue')

    sns.scatterplot(x=[min_risk_portfolio_std], y=[min_risk_portfolio_return], marker='.', s=500, label=f'Min. risk portfolio, {mes_short_sales}', color=color)

    sns.lineplot(x=portfolio_stds, y=portfolio_returns, label=f'Efficient frontier of {num}, {mes_short_sales}', color=color)

    plt.legend()
    plt.ylabel('Expected return')
    plt.xlabel('Standard deviation')