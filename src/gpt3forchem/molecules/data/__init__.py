import pandas as pd
import os

THIS_DIR = os.path.abspath(os.path.dirname(__file__))


def get_data():
    return pd.read_csv(os.path.join(THIS_DIR, "data_comp.csv"))
