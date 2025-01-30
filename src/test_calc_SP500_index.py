from pathlib import Path

import numpy as np
import pandas as pd

import calc_SP500_index
import pull_CRSP_stock
import pull_SP500_constituents
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
START_DATE = calc_SP500_index.START_DATE
END_DATE = calc_SP500_index.END_DATE

years = (END_DATE - START_DATE).days / 365.25

df_constituents = pull_SP500_constituents.load_constituents(data_dir=DATA_DIR)
df_msf = pull_CRSP_stock.load_CRSP_monthly_file(data_dir=DATA_DIR)
df_msix = pull_CRSP_stock.load_CRSP_index_files(data_dir=DATA_DIR)

sp500_total_market_cap = calc_SP500_index.calculate_sp500_total_market_cap(
    df_constituents, df_msf
)

sp500_total_market_cap_and_returns = (
    calc_SP500_index.append_actual_sp500_index_and_approx_returns_A(
        sp500_total_market_cap, df_msix
    )
)
sp500_data = calc_SP500_index.create_sp500_index_approximations(data_dir=DATA_DIR)


def test_date_range():
    """Test that the date range matches expected values"""
    # Check start and end dates
    assert sp500_data["date"].min() == pd.Timestamp("1990-01-31")
    assert sp500_data["date"].max() == pd.Timestamp("2022-12-30")

    # Check number of observations
    assert len(sp500_data) == 396


def test_constituent_counts():
    """Test that constituent counts are within expected ranges"""
    counts = sp500_data["n_constituents"].describe()

    assert counts["min"] >= 441
    assert counts["max"] <= 493


def test_return_correlations_A():
    """Test that return approximation A is highly correlated with the actual
    returns on the S&P500"""
    returns = sp500_data[["sprtrn", "ret_approx_A"]]
    corr = returns.corr()

    # Expected minimum correlations based on provided results
    min_corr_A = 0.992116  # correlation between sprtrn and ret_approx_A

    assert corr.loc["sprtrn", "ret_approx_A"] >= min_corr_A


def test_return_correlations_B():
    """Test that return approximation B is highly correlated with the actual
    returns on the S&P500"""
    returns = sp500_data[["sprtrn", "ret_approx_B"]]
    corr = returns.corr()

    # Expected minimum correlations based on provided results
    min_corr_B = 0.9987  # correlation between sprtrn and ret_approx_B

    assert corr.loc["sprtrn", "ret_approx_B"] >= min_corr_B


def test_mean_returns_approx_A():
    """Test that mean returns are close to each other"""

    assert (
        np.abs(sp500_data["ret_approx_A"].mean() - sp500_data["sprtrn"].mean()) < 0.0012
    )


def test_mean_returns_approx_B():
    """Test that mean returns are close to each other"""

    assert (
        np.abs(sp500_data["ret_approx_B"].mean() - sp500_data["sprtrn"].mean())
        < 0.00025
    )


def test_mean_returns_sprtrn():
    """Test that mean S&P500 return is correct"""

    assert np.abs(sp500_data["sprtrn"].mean() - 0.006457) < 0.001


def test_market_cap_normalization():
    """Test that market cap normalization is done correctly"""
    # First value of normalized market cap should equal first value of index
    first_index = sp500_data["spindx"].iloc[0]
    first_norm_cap = sp500_data["sp500_market_cap_norm"].iloc[0]
    assert np.isclose(first_index, first_norm_cap, rtol=1e-3)


def test_cumulative_returns():
    """Test that cumulative returns are calculated correctly"""
    # Test that cumulative returns are monotonically increasing over the long term
    assert (
        sp500_data["cumret_approx_A"].iloc[-10] > sp500_data["cumret_approx_A"].iloc[-1]
    )
    assert sp500_data["sp500_cumret"].iloc[-10] > sp500_data["sp500_cumret"].iloc[-1]

    # Test final cumulative return magnitudes

    assert 0.6 < sp500_data["cumret_approx_A"].max()/years < 0.65
    assert 0.35 < sp500_data["sp500_cumret"].max()/years < 0.4


def test_constituent_columns():
    """Test that constituent DataFrame has required columns"""
    required_columns = {"permno", "indno", "mbrstartdt", "mbrenddt", "mbrflg", "indfam"}
    actual_columns = set(df_constituents.columns)
    assert required_columns.issubset(actual_columns), (
        f"Missing columns in constituents: {required_columns - actual_columns}"
    )


def test_msf_columns():
    """Test that monthly stock file DataFrame has required columns"""
    required_columns = {
        "date",
        "permno",
        "shrcd",
        "exchcd",
        "comnam",
        "shrcls",
        "ret",
        "retx",
        "dlret",
        "dlretx",
        "dlstcd",
        "prc",
        "shrout",
        "cfacshr",
        "cfacpr",
        "adj_shrout",
    }
    actual_columns = set(df_msf.columns)
    assert required_columns.issubset(actual_columns), (
        f"Missing columns in msf: {required_columns - actual_columns}"
    )


def test_msix_columns():
    """Test that index file DataFrame has required columns"""
    required_columns = {
        "caldt",
        "vwretd",
        "vwindd",
        "vwretx",
        "vwindx",
        "ewretd",
        "ewindd",
        "ewretx",
        "ewindx",
        "sprtrn",
        "spindx",
    }
    actual_columns = set(df_msix.columns)
    assert required_columns.issubset(actual_columns), (
        f"Missing columns in msix: {required_columns - actual_columns}"
    )


def test_market_cap_columns():
    """Test that total market cap DataFrame has required columns"""
    df_constituents = pull_SP500_constituents.load_constituents(data_dir=DATA_DIR)
    df_msf = pull_CRSP_stock.load_CRSP_monthly_file(data_dir=DATA_DIR)
    sp500_total_market_cap = calc_SP500_index.calculate_sp500_total_market_cap(
        df_constituents, df_msf
    )
    required_columns = {"date", "sp500_market_cap", "n_constituents"}
    actual_columns = set(sp500_total_market_cap.columns)
    assert required_columns.issubset(actual_columns), (
        f"Missing columns in total market cap: {required_columns - actual_columns}"
    )


def test_market_cap_and_returns_columns():
    """Test that market cap and returns DataFrame has required columns"""
    df_constituents = pull_SP500_constituents.load_constituents(data_dir=DATA_DIR)
    df_msf = pull_CRSP_stock.load_CRSP_monthly_file(data_dir=DATA_DIR)
    df_msix = pull_CRSP_stock.load_CRSP_index_files(data_dir=DATA_DIR)

    sp500_total_market_cap = calc_SP500_index.calculate_sp500_total_market_cap(
        df_constituents, df_msf
    )
    sp500_total_market_cap_and_returns = (
        calc_SP500_index.append_actual_sp500_index_and_approx_returns_A(
            sp500_total_market_cap, df_msix
        )
    )

    required_columns = {
        "date",
        "spindx",
        "sprtrn",
        "sp500_market_cap",
        "n_constituents",
        "sp500_market_cap_norm",
        "ret_approx_A",
        "cumret_approx_A",
        "sp500_cumret",
    }
    actual_columns = set(sp500_total_market_cap_and_returns.columns)
    assert required_columns.issubset(actual_columns), (
        f"Missing columns in market cap and returns: {required_columns - actual_columns}"
    )


def test_final_data_columns():
    """Test that final DataFrame has all required columns"""
    sp500_data = calc_SP500_index.create_sp500_index_approximations(data_dir=DATA_DIR)
    required_columns = {
        "date",
        "spindx",
        "sprtrn",
        "sp500_market_cap",
        "n_constituents",
        "sp500_market_cap_norm",
        "ret_approx_A",
        "cumret_approx_A",
        "sp500_cumret",
        "ret_approx_B",
    }
    actual_columns = set(sp500_data.columns)
    assert required_columns.issubset(actual_columns), (
        f"Missing columns in final data: {required_columns - actual_columns}"
    )
