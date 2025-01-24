"""
Functions to pull and calculate the value and equal weighted CRSP indices.

 - Data for indices: https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_a_indexes/
 - Data for raw stock data: https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/annual-update/stock-security-files/monthly-stock-file/
 - Why we can't perfectly replicate them: https://wrds-www.wharton.upenn.edu/pages/support/support-articles/crsp/index-and-deciles/constructing-value-weighted-return-series-matches-vwretd-crsp-monthly-value-weighted-returns-includes-distributions/
 - Methodology used: https://wrds-www.wharton.upenn.edu/documents/396/CRSP_US_Stock_Indices_Data_Descriptions.pdf
 - Useful link: https://www.tidy-finance.org/python/wrds-crsp-and-compustat.html

Thank you to Tobias Rodriguez del Pozo for his assistance in writing this
code.

Note: This code is based on the old CRSP SIZ format. Information
about the new CIZ format can be found here:

 - Transition FAQ: https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/stocks-and-indices/crsp-stock-and-indexes-version-2/crsp-ciz-faq/
 - CRSP Metadata Guide: https://wrds-www.wharton.upenn.edu/documents/1941/CRSP_METADATA_GUIDE_STOCK_INDEXES_FLAT_FILE_FORMAT_2_0_CIZ_09232022v.pdf

"""

from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

import numpy as np
import pandas as pd
import wrds

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")


def pull_CRSP_monthly_file(
    start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME
):
    """
    Pulls monthly CRSP stock data from a specified start date to end date.

    SQL query to pull data, controls for delisting, and importantly
    follows the guidelines that CRSP uses for inclusion, with the exception
    of code 73, which is foreign companies -- without including this, the universe
    of securities is roughly half of what it should be.

    Notes
    -----

    From Bali, Engle, Murray -- Empirical asset pricing-the cross section of stock returns (2016)
    they say,
    "There are two main proxies for the market portfolio that are commonly used
    in empirical asset pricing research. The first is the value-weighted portfolio of all
    U.S.-based common stocks in the CRSP database. The daily and monthly excess
    returns for this portfolio are available from Ken French’s website, as well as from
    the Fama–French database on WRDS. We follow common convention by referring
    to this portfolio and its excess returns as MKT, which stands for market minus
    risk-free. The second portfolio commonly used as a proxy for the market portfolio
    is the CRSP value-weighted portfolio, which contains all securities in the CRSP
    database, not just common stocks, but excluding American Depository Receipts
    (ADRs).11 Following CRSP, we denote this portfolio VWRETD. The main difference
    between the VWRETD portfolio and the MKT portfolio is that the VWRETD portfolio
    contains shares of firms that are not based in the United States, closed-end funds,
    and other securities that are not common stocks. Daily and monthly returns for this
    portfolio are available from CRSP.
    "


    **Share Code Meaning:**

    * **10, 11:** Common stock
    * **20, 21:** Preferred stock
    * **40, 41:** Warrants
    * **70, 71:** Units
    * **73:** Foreign stocks

    **Why Subset to These Share Codes?**

    By filtering for these share codes, you're typically focusing on securities
    that are more widely traded and have a higher level of liquidity. This is
    because:

    1. **Common Stock:** These are the most common type of equity security and
       are often the most liquid.
    2. **Preferred Stock:** While less liquid than common stock, preferred stock
       can still be a valuable component of an investment portfolio.
    3. **Warrants:** These are options to buy a specific number of shares of a
       company's stock at a fixed price. They can be volatile but offer
       potential for high returns.
    4. **Units:** These are often associated with real estate investment trusts
       (REITs) and can provide exposure to real estate markets.
    5. **Foreign Stocks:** Including foreign stocks can broaden your investment
       universe and provide exposure to international markets.

    **Why Exclude Other Share Codes?**

    Other share codes, such as those for closed-end funds, exchange-traded funds
    (ETFs), and depositary receipts, may be excluded because:

    * **Less Liquid:** These securities may be less liquid than common stock and
      can be more difficult to trade.
    * **Different Risk Profiles:** These securities may have different risk
      profiles than common stock, making them less suitable for certain
      investment strategies.
    * **Complex Structures:** Some of these securities have complex structures
      that can make them difficult to analyze.

    By focusing on the specified share codes, you can create a more focused and
    investable universe of stocks. However, the optimal choice of share codes
    will depend on your specific investment goals and risk tolerance.

    """
    # Convert start_date to datetime if it's a string
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    # Not a perfect solution, but since value requires t-1 period market cap,
    # we need to pull one extra month of data. This is hidden from the user.
    start_date = start_date - relativedelta(months=1)
    start_date = start_date.strftime("%Y-%m-%d")

    query = f"""
    SELECT 
        date,
        msf.permno, msf.permco, shrcd, exchcd, comnam, shrcls, 
        ret, retx, dlret, dlretx, dlstcd,
        prc, altprc, vol, shrout, cfacshr, cfacpr,
        naics, siccd
    FROM crsp.msf AS msf
    LEFT JOIN 
        crsp.msenames as msenames
    ON 
        msf.permno = msenames.permno AND
        msenames.namedt <= msf.date AND
        msf.date <= msenames.nameendt
    LEFT JOIN 
        crsp.msedelist as msedelist
    ON 
        msf.permno = msedelist.permno AND
        date_trunc('month', msf.date)::date =
        date_trunc('month', msedelist.dlstdt)::date
    WHERE 
        msf.date BETWEEN '{start_date}' AND '{end_date}' AND 
        msenames.shrcd IN (10, 11, 20, 21, 40, 41, 70, 71, 73)
    """
    # with wrds.Connection(wrds_username=wrds_username) as db:
    #     df = db.raw_sql(
    #         query, date_cols=["date", "namedt", "nameendt", "dlstdt"]
    #     )
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["date", "namedt", "nameendt", "dlstdt"])
    db.close()

    df = df.loc[:, ~df.columns.duplicated()]
    df["shrout"] = df["shrout"] * 1000

    # Also, as an additional note, CRSP reports that "cfacshr" and "cfacpr" are
    # not always equal. This means that we cannot use `market_cap` = `prc` *
    # `shrout` alone. We need to use the cumulative adjustment factors to adjust
    # for corporate actions that affect the stock price, such as stock splits.
    # "cfacshr" and "cfacpr" are not always equal because of less common
    # distribution events, spinoffs, and rights. See here: [CRSP - Useful
    # Variables](https://vimeo.com/443061703)

    df["adj_shrout"] = df["shrout"] * df["cfacshr"]
    df["adj_prc"] = df["prc"].abs() / df["cfacpr"]
    df["market_cap"] = df["adj_prc"] * df["adj_shrout"]

    # Deal with delisting returns
    df = apply_delisting_returns(df)

    return df


def apply_delisting_returns(df):
    """
    Use instructions for handling delisting returns from: Chapter 7 of
    Bali, Engle, Murray --
    Empirical asset pricing-the cross section of stock returns (2016)

    First change dlret column.
    If dlret is NA and dlstcd is not NA, then:
    if dlstcd is 500, 520, 551-574, 580, or 584, then dlret = -0.3
    if dlret is NA but dlstcd is not one of the above, then dlret = -1
    """
    df["dlret"] = np.select(
        [
            df["dlstcd"].isin([500, 520, 580, 584] + list(range(551, 575)))
            & df["dlret"].isna(),
            df["dlret"].isna() & df["dlstcd"].notna() & df["dlstcd"] >= 200,
            True,
        ],
        [-0.3, -1, df["dlret"]],
        default=df["dlret"],
    )

    df["dlretx"] = np.select(
        [
            df["dlstcd"].isin([500, 520, 580, 584] + list(range(551, 575)))
            & df["dlretx"].isna(),
            df["dlretx"].isna() & df["dlstcd"].notna() & df["dlstcd"] >= 200,
            True,
        ],
        [-0.3, -1, df["dlretx"]],
        default=df["dlretx"],
    )

    # Replace the inplace operations with direct assignments
    df["ret"] = df["ret"].fillna(df["dlret"])
    df["retx"] = df["retx"].fillna(df["dlretx"])
    return df


def apply_delisting_returns_alt(df):
    df["dlret"] = df["dlret"].fillna(0)
    df["ret"] = df["ret"] + df["dlret"]
    df["ret"] = np.where(
        (df["ret"].isna()) & (df["dlret"] != 0), df["dlret"], df["ret"]
    )
    return df


def pull_CRSP_index_files(
    start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME
):
    """
    Pulls the CRSP index files from crsp_a_indexes.msix:
    (Monthly)NYSE/AMEX/NASDAQ Capitalization Deciles, Annual Rebalanced (msix)
    """
    # Pull index files
    query = f"""
        SELECT * 
        FROM crsp_a_indexes.msix
        WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
    """
    # with wrds.Connection(wrds_username=wrds_username) as db:
    #     df = db.raw_sql(query, date_cols=["month", "caldt"])
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["caldt"])
    db.close()
    return df


def load_CRSP_monthly_file(data_dir=DATA_DIR):
    path = Path(data_dir) / "CRSP_MSF_INDEX_INPUTS.parquet"
    df = pd.read_parquet(path)
    return df


def load_CRSP_index_files(data_dir=DATA_DIR):
    path = Path(data_dir) / f"CRSP_MSIX.parquet"
    df = pd.read_parquet(path)
    return df


def _demo():
    df_msf = load_CRSP_monthly_file(data_dir=DATA_DIR)
    df_msix = load_CRSP_index_files(data_dir=DATA_DIR)


if __name__ == "__main__":
    df_msf = pull_CRSP_monthly_file(start_date=START_DATE, end_date=END_DATE)
    path = Path(DATA_DIR) / "CRSP_MSF_INDEX_INPUTS.parquet"
    df_msf.to_parquet(path)

    df_msix = pull_CRSP_index_files(start_date=START_DATE, end_date=END_DATE)
    path = Path(DATA_DIR) / f"CRSP_MSIX.parquet"
    df_msix.to_parquet(path)
