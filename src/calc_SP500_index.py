import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import pull_CRSP_stock
import pull_SP500_constituents
from settings import config

DATA_DIR = config("DATA_DIR")
START_DATE = pd.to_datetime("1990-01-31")
END_DATE = pd.to_datetime("2023-12-29")


def calculate_sp500_total_market_cap(
    df_constituents, df_msf, start_date=START_DATE, end_date=END_DATE
):
    """
    Calculate total market capitalization of S&P 500 constituents for each month
    """
    # Convert dates
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter CRSP data to date range
    df_msf = df_msf[
        (df_msf["date"] >= start_date) & (df_msf["date"] <= end_date)
    ].copy()


    # Calculate market cap for each stock, making sure to use CRSP's
    # cumulative factors for shares and prices to adjust for splits and
    # dividends.
    df_msf["adj_shrout"] = df_msf["shrout"] * df_msf["cfacshr"]
    df_msf["adj_prc"] = df_msf["prc"].abs() / df_msf["cfacpr"]
    df_msf["market_cap"] = df_msf["adj_prc"] * df_msf["adj_shrout"]

    # Pre-allocate arrays for results
    dates = df_msf["date"].drop_duplicates().sort_values().to_numpy()
    n_dates = len(dates)
    sp500_market_cap = np.zeros(n_dates)
    n_constituents = np.zeros(n_dates, dtype=int)

    # Calculate for each month-end date
    for i, date in enumerate(dates):
        ## YOUR CODE HERE
        pass
        ## YOUR CODE HERE

    # Create DataFrame from arrays
    results_df = pd.DataFrame(
        {
            "date": dates,
            "sp500_market_cap": sp500_market_cap,
            "n_constituents": n_constituents,
        }
    )

    return results_df


def append_actual_sp500_index_and_approx_returns_A(sp500_total_market_cap, df_msix):
    """
    Append the actual S&P 500 index level and returns to the sp500_total_market_cap DataFrame.
    Then, create a normalized market cap series. This normalized market cap series is
    the total market cap series calculated in `calculate_sp500_total_market_cap`,
    normalized so that the first value is equal to the first value of the S&P 500 index.

    Takes in the DataFrame from `calculate_sp500_total_market_cap` and the DataFrame
    from `pull_CRSP_stock.load_CRSP_index_files`.

    Returns the same DataFrame as `sp500_total_market_cap`, but with the following
    additional columns:

      - `spindx`: the actual S&P 500 index level
      - `sprtrn`: the actual S&P 500 index returns
      - `sp500_market_cap_norm`: the normalized market cap series
      - `ret_approx_A`: the simple returns of the normalized market cap series
      - `cumret_approx_A`: the cumulative returns of the normalized market cap series
      - `sp500_cumret`: the cumulative returns of the actual S&P 500 index

    """
    # Merge in the actual S&P 500 index level and returns
    df_msix["date"] = pd.to_datetime(df_msix["caldt"])
    sp500_total_market_cap = pd.merge(
        df_msix[["date", "spindx", "sprtrn"]],
        sp500_total_market_cap,
        on="date",
        how="inner",
    )

    # Create normalized market cap series
    ## YOUR CODE HERE

    # Calculate simple returns from market cap changes
    sp500_total_market_cap["ret_approx_A"] = sp500_total_market_cap[
        "sp500_market_cap"
    ].pct_change()

    # Calculate cumulative returns (adding 1 to start from 100%)
    sp500_total_market_cap["cumret_approx_A"] = (
        1 + sp500_total_market_cap["ret_approx_A"]
    ).cumprod()
    sp500_total_market_cap["sp500_cumret"] = (
        1 + sp500_total_market_cap["sprtrn"]
    ).cumprod()


    return sp500_total_market_cap




def is_rebalance_month(date):
    """
    Check if the given date is in a rebalancing month (Mar, Jun, Sep, Dec)

    Args:
        date: pandas Timestamp or datetime object
    Returns:
        bool: True if date is in a rebalancing month
    """
    return date.month in [3, 6, 9, 12]


def calculate_sp500_returns_with_rebalancing(
    df_constituents, df_msf, start_date=START_DATE, end_date=END_DATE
):
    """
    Calculate S&P 500 returns with rebalancing.

    Takes in the DataFrame from `pull_SP500_constituents.load_constituents` and the
    DataFrame from `pull_CRSP_stock.load_CRSP_monthly_file`.

    Returns a DataFrame with the following columns:
      - `date`: the date
      - `ret_approx_B`: the simple returns of the S&P 500 index using the approximation B
    """
    # Convert dates
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter CRSP data to date range
    df_msf = df_msf[
        (df_msf["date"] >= start_date) & (df_msf["date"] <= end_date)
    ].copy()

    # Calculate market cap for each stock
    df_msf["market_cap"] = abs(df_msf["prc"]) * df_msf["shrout"]

    # Initialize weights DataFrame with zeros
    dates = df_msf["date"].drop_duplicates().sort_values().reset_index(drop=True)
    all_permno = df_msf["permno"].drop_duplicates().sort_values()
    sp500_weights = pd.DataFrame(0.0, index=dates, columns=all_permno)
    portfolio_weights = pd.DataFrame(0.0, index=dates, columns=all_permno)
    sp500_returns = pd.DataFrame(np.nan, index=dates, columns=["ret_approx_B"])

    # Calculate weights for each date
    for i, date in enumerate(dates):
        ## YOUR CODE HERE
        pass
        ## YOUR CODE HERE

    # # Verify the weights sum to approximately 1 for each date
    weight_sums = sp500_weights.sum(axis=1)
    # assert np.allclose(weight_sums, 1.0, atol=1e-6)

    # Ensure both matrices are sorted the same way
    ret_matrix = (
        df_msf[["date", "permno", "retx"]]
        .set_index(["date", "permno"])
        .unstack()["retx"]
        .sort_index(axis=0)  # Sort dates
        .sort_index(axis=1)  # Sort permnos
    )
    portfolio_weights = portfolio_weights.sort_index(axis=0).sort_index(axis=1)

    # Shift weights up by one period (so t+1 returns multiply with t weights)
    lagged_weights = portfolio_weights.shift(1)

    # Compute returns for each date using matrix multiplication
    sp500_returns["ret_approx_B"] = (ret_matrix * lagged_weights).sum(
        axis=1, skipna=True
    )
    sp500_returns.iloc[0, 0] = np.nan
    sp500_returns = sp500_returns.reset_index()
    return sp500_returns


def _demo_approximation_A():
    """
    Calculate the S&P 500 index using the approximation A. That is,
    simply sum the market cap of the stocks in the S&P 500. Normalize that
    series so that the first value is equal to the first value of the S&P 500 index.

    Also, approximate the returns of the S&P 500 index by using the simple returns
    of the normalized market cap series.
    """
    df_constituents = pull_SP500_constituents.load_constituents(data_dir=DATA_DIR)
    df_msf = pull_CRSP_stock.load_CRSP_monthly_file(data_dir=DATA_DIR)
    df_msix = pull_CRSP_stock.load_CRSP_index_files(data_dir=DATA_DIR)

    sp500_total_market_cap = calculate_sp500_total_market_cap(
        df_constituents, df_msf, start_date=START_DATE, end_date=END_DATE
    )

    sp500_total_market_cap = append_actual_sp500_index_and_approx_returns_A(
        sp500_total_market_cap, df_msix
    )

    if True:
        # Plot both series
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=sp500_total_market_cap,
            x="date",
            y="sp500_market_cap_norm",
            label="Normalized Market Cap",
        )
        sns.lineplot(
            data=sp500_total_market_cap, x="date", y="spindx", label="S&P 500 Index"
        )

        plt.title("S&P 500: Index Level vs Normalized Market Cap")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.show()

    # Print correlation
    correlation = sp500_total_market_cap["sp500_market_cap_norm"].corr(
        sp500_total_market_cap["spindx"]
    )
    print(f"Correlation between normalized market cap and index: {correlation:.4f}")

    if True:
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=sp500_total_market_cap,
            x="date",
            y="cumret_approx_A",
            label="Cumulative Return (Market Cap)",
        )
        sns.lineplot(
            data=sp500_total_market_cap,
            x="date",
            y="sp500_cumret",
            label="Cumulative Return (S&P 500)",
        )

        plt.title("S&P 500: Cumulative Returns Comparison")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return (1 = Initial Value)")
        plt.grid(True)
        plt.legend()
        plt.show()

    # Print correlation of returns
    correlation = sp500_total_market_cap["ret_approx_A"].corr(
        sp500_total_market_cap["sprtrn"]
    )
    print(f"Correlation between returns: {correlation:.4f}")


def _demo_approximation_B():
    """
    Calculate the S&P 500 index using the approximation B. That is,
    rebalance the portfolio every quarter.
    """
    df_constituents = pull_SP500_constituents.load_constituents(data_dir=DATA_DIR)
    df_msf = pull_CRSP_stock.load_CRSP_monthly_file(data_dir=DATA_DIR)
    df_msix = pull_CRSP_stock.load_CRSP_index_files(data_dir=DATA_DIR)

    sp500_returns = calculate_sp500_returns_with_rebalancing(
        df_constituents, df_msf, start_date=START_DATE, end_date=END_DATE
    )
    df_msix["date"] = pd.to_datetime(df_msix["caldt"])
    sp500_returns = pd.merge(
        df_msix[["date", "spindx", "sprtrn"]],
        sp500_returns,
        on="date",
        how="inner",
    )

    sp500_returns["diff"] = sp500_returns["ret_approx_B"] - sp500_returns["sprtrn"]
    sns.lineplot(data=sp500_returns, x="date", y="diff", label="Rebalanced Portfolio")
    sp500_returns.describe()

    # Print correlation
    correlation = sp500_returns["ret_approx_B"].corr(sp500_returns["sprtrn"])
    print(f"Correlation between reconstructed and actual returns: {correlation:.4f}")

    # Calculate cumulative returns
    sp500_returns["cumret_approx_B"] = (1 + sp500_returns["ret_approx_B"]).cumprod()
    sp500_returns["cumret_actual"] = (1 + sp500_returns["sprtrn"]).cumprod()

    if True:
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=sp500_returns,
            x="date",
            y="cumret_approx_B",
            label="Cumulative Return (Rebalanced Portfolio)",
        )
        sns.lineplot(
            data=sp500_returns,
            x="date",
            y="cumret_actual",
            label="Cumulative Return (S&P 500)",
        )
        plt.title("S&P 500: Cumulative Returns with Quarterly Rebalancing")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return (1 = Initial Value)")
        plt.grid(True)
        plt.legend()
        plt.show()

    # Print correlation of returns
    correlation = sp500_returns["ret_approx_B"].corr(sp500_returns["sprtrn"])
    print(f"Correlation between returns: {correlation:.4f}")

    df_constituents = pull_SP500_constituents.load_constituents(data_dir=DATA_DIR)
    df_msf = pull_CRSP_stock.load_CRSP_monthly_file(data_dir=DATA_DIR)
    df_msix = pull_CRSP_stock.load_CRSP_index_files(data_dir=DATA_DIR)

    sp500_total_market_cap = calculate_sp500_total_market_cap(
        df_constituents, df_msf, start_date=START_DATE, end_date=END_DATE
    )

    sp500_total_market_cap = append_actual_sp500_index_and_approx_returns_A(
        sp500_total_market_cap, df_msix
    )

    sp500_returns = pd.merge(
        sp500_returns,
        sp500_total_market_cap[["date", "ret_approx_A"]],
        on="date",
        how="inner",
    )

    sp500_returns["diff_A_less_B"] = (
        sp500_returns["ret_approx_A"] - sp500_returns["ret_approx_B"]
    )
    sns.lineplot(
        data=sp500_returns, x="date", y="diff_A_less_B", label="Reconstructed Portfolio"
    )

    sp500_returns["diff_A_less_B"].describe()


def create_sp500_index_approximations(data_dir=DATA_DIR):
    df_constituents = pull_SP500_constituents.load_constituents(data_dir=data_dir)
    df_msf = pull_CRSP_stock.load_CRSP_monthly_file(data_dir=data_dir)
    df_msix = pull_CRSP_stock.load_CRSP_index_files(data_dir=data_dir)

    ## Approximation A
    sp500_total_market_cap = calculate_sp500_total_market_cap(
        df_constituents, df_msf, start_date=START_DATE, end_date=END_DATE
    )

    sp500_total_market_cap = append_actual_sp500_index_and_approx_returns_A(
        sp500_total_market_cap,
        df_msix,
    )

    ## Approximation B
    sp500_returns = calculate_sp500_returns_with_rebalancing(
        df_constituents, df_msf, start_date=START_DATE, end_date=END_DATE
    )

    df = pd.merge(sp500_total_market_cap, sp500_returns, on="date", how="inner")

    # df.info()
    # df[["sprtrn", "ret_approx_A", "ret_approx_B"]].corr()

    return df


if __name__ == "__main__":
    df = create_sp500_index_approximations(data_dir=DATA_DIR)
    df.to_parquet(DATA_DIR / "sp500_index_approximations.parquet")
