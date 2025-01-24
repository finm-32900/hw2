import pandas as pd

import pull_SP500_constituents
from settings import config

DATA_DIR = config("DATA_DIR")


def test_load_constituents():
    df_constituents = pull_SP500_constituents.load_constituents(data_dir=DATA_DIR)
    example_1 = df_constituents[df_constituents["permno"] == 93159]
    assert example_1["mbrstartdt"].values[0] == pd.to_datetime("2012-07-31")
    assert example_1["mbrenddt"].values[0] == pd.to_datetime("2016-03-29")

    example_2 = df_constituents[df_constituents["permno"] == 10078]
    assert example_2["mbrstartdt"].values[0] == pd.to_datetime("1992-08-20")
    assert example_2["mbrenddt"].values[0] == pd.to_datetime("2010-01-28")
