import pandas as pd
import wrds

from settings import config

DATA_DIR = config("DATA_DIR")
WRDS_USERNAME = config("WRDS_USERNAME")


def pull_constituents(wrds_username=WRDS_USERNAME):
    db = wrds.Connection(wrds_username=wrds_username)

    df_constituents = db.raw_sql(""" 
    ## YOUR CODE HERE
    from crsp_m_indexes.dsp500list_v2 
    """)

    # Convert string columns to datetime if they aren't already
    df_constituents["mbrstartdt"] = pd.to_datetime(df_constituents["mbrstartdt"])
    df_constituents["mbrenddt"] = pd.to_datetime(df_constituents["mbrenddt"])

    return df_constituents


def load_constituents(data_dir=DATA_DIR):
    return pd.read_parquet(data_dir / "df_sp500_constituents.parquet")


def _demo():
    df_constituents = pull_constituents(wrds_username=WRDS_USERNAME)
    df_constituents.describe()

    import pull_CRSP_stock

    # df_msf = pull_CRSP_stock.load_CRSP_monthly_file(data_dir=DATA_DIR)
    df_msix = pull_CRSP_stock.load_CRSP_index_files(data_dir=DATA_DIR)

    df_msix.describe()
    df_msix.info()
    df_msix[["caldt", "sprtrn", "spindx"]].describe()


if __name__ == "__main__":
    df_constituents = pull_constituents(wrds_username=WRDS_USERNAME)
    df_constituents.to_parquet(DATA_DIR / "df_sp500_constituents.parquet")
