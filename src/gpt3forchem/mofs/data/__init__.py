import os
import pandas as pd

THIS_DIR = os.path.abspath(os.path.dirname(__file__))


def get_data():
    df = pd.read_csv(os.path.join(THIS_DIR, "data.csv"))
    good_rows = []
    for i, row in df.iterrows():
        if not "ERROR" in row["mofid.mofid"]:
            if not "UNKNOWN" in row["mofid.mofid"]:
                # didn't have this last check in the experiments
                if not "NA." in row["mofid.mofid"]:
                    good_rows.append(row)
    return pd.DataFrame(good_rows)
