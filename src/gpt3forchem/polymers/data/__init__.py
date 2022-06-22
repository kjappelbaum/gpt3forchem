import os
import pandas as pd

THIS_DIR = os.abspath(os.path.dirname(__file__))


def get_data():
    return pd.read_csv(os.path.join(THIS_DIR, "data.csv"))
