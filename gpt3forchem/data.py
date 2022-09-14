# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/00_data.ipynb.

# %% auto 0
__all__ = ['POLYMER_FEATURES', 'gas_features', 'discretize', 'get_polymer_data', 'get_photoswitch_data', 'get_mof_data',
           'preprocess_mof_data', 'get_core_mof_data', 'get_opv_data']

# %% ../notebooks/00_data.ipynb 2
import os
from collections import Counter

import pandas as pd

from .helpers import HashableDataFrame

_THIS_DIR = os.path.abspath(os.path.dirname(os.path.abspath("")))
import numpy as np 


# %% ../notebooks/00_data.ipynb 4
def discretize(
    df: pd.DataFrame, col: str, n_bins: int = 5, new_name: str = None, labels=None
) -> None:
    """Adds a new column to the dataframe with the discretized values of the column."""
    if new_name is None:
        new_name = col + "_cat"
    if labels is None:
        labels = ["very small", "small", "medium", "large", "very large"]

    df[new_name] = pd.cut(
        df[col],
        n_bins,
        labels=labels,
    )


# %% ../notebooks/00_data.ipynb 6
POLYMER_FEATURES = [
    "num_[W]",
    "max_[W]",
    "num_[Tr]",
    "max_[Tr]",
    "num_[Ta]",
    "max_[Ta]",
    "num_[R]",
    "max_[R]",
    "[W]",
    "[Tr]",
    "[Ta]",
    "[R]",
    "rel_shannon",
    "length",
]


# %% ../notebooks/00_data.ipynb 7
def get_polymer_data(datadir="../data"):  # path to folder with data files
    return HashableDataFrame(pd.read_csv(os.path.join(datadir, "polymers.csv")))


# %% ../notebooks/00_data.ipynb 13
def get_photoswitch_data(datadir="../data"):  # path to folder with data files
    """By default we drop the rows without E isomer pi-pi* transition wavelength."""
    df = pd.read_csv(os.path.join(datadir, "photoswitches.csv"))
    df.dropna(subset=["E isomer pi-pi* wavelength in nm"], inplace=True)
    df.drop_duplicates(
        subset=["SMILES"], inplace=True
    )  # not sure how and if they did this in the initial work. There are certainly duplicates, e.g. C[N]1C=CC(=N1)N=NC2=CC=CC=C2 (see top)
    df.reset_index(inplace=True)
    return HashableDataFrame(df)


# %% ../notebooks/00_data.ipynb 26
def get_mof_data(datadir="../data"):  # path to folder with data files
    df =  HashableDataFrame(pd.read_csv(os.path.join(datadir, "mof.csv")))


    return df

# %% ../notebooks/00_data.ipynb 28
def preprocess_mof_data(mof_data, n_bins=None, labels=None): 
    if n_bins is None:
        n_bins = 3
    if labels is None:
        labels = ["low", "medium", "high"]
    features = [
        "outputs.Xe-henry_coefficient-mol--kg--Pa",
        "outputs.Kr-henry_coefficient-mol--kg--Pa",
        "outputs.H2O-henry_coefficient-mol--kg--Pa",
        "outputs.H2S-henry_coefficient-mol--kg--Pa",
        "outputs.CO2-henry_coefficient-mol--kg--Pa",
        "outputs.CH4-henry_coefficient-mol--kg--Pa",
        "outputs.O2-henry_coefficient-mol--kg--Pa",
    ]

    for feature in features:
        mof_data[feature + '_log'] = np.log10(mof_data[feature] + 1e-40)

    for feature in features:

        discretize(
            mof_data, f"{feature}_log", n_bins=n_bins, labels=labels
        )

# %% ../notebooks/00_data.ipynb 37
def get_core_mof_data(datadir="../data"):  # path to folder with data files
    return HashableDataFrame(pd.read_csv(os.path.join(datadir, "core_mof.csv")))


# %% ../notebooks/00_data.ipynb 40
gas_features = pd.DataFrame(
    [
        {
            "name": "carbon_dioxide",
            "formula": "CO2",
            "critical_temperature": 304.19,
            "critical_pressure": 7382000,
            "accentric_factor": 0.228,
            "radius": 1.525,
            "polar": False,
            "related_column": "outputs.CO2-henry_coefficient-mol--kg--Pa_log_cat"
        },
        {
            "name": "xenon",
            "formula": "Xe",
            "critical_temperature": 289.74,
            "critical_pressure": 5840000,
            "accentric_factor": 0,
            "radius": 1.985,
            "polar": False,
            "related_column": "outputs.Xe-henry_coefficient-mol--kg--Pa_log_cat"
        },
        {
            "name": "krypton",
            "formula": "Kr",
            "critical_temperature": 209.35,
            "critical_pressure": 5502000,
            "accentric_factor": 0,
            "radius": 1.83,
            "polar": False,
            "related_column": "outputs.Kr-henry_coefficient-mol--kg--Pa_log_cat"
        },
        {
            "name": "hydrogen disulfide",
            "formula": "H2S",
            "critical_temperature": 373.53,
            "critical_pressure": 8963000,
            "accentric_factor": 0.0942,
            "radius": 1.74,
            "polar": True,
            "related_column": "outputs.H2S-henry_coefficient-mol--kg--Pa_log_cat"
        },
        {
            "name": "water",
            "formula": "H2O",
            "critical_temperature": 647.16,
            "critical_pressure": 22055000,
            "accentric_factor": 0.3449,
            "radius": 1.58,
            "polar": True,
            "related_column": "outputs.H2O-henry_coefficient-mol--kg--Pa_log_cat"
        },
        {
            "name": "methane",
            "formula": "CH4",
            "critical_temperature": 190.56,
            "critical_pressure": 4599000,
            "accentric_factor": 0.012,
            "radius": 1.865,
            "polar": False,
            "related_column": "outputs.CH4-henry_coefficient-mol--kg--Pa_log_cat"
        },
        {
            "name": "oxygen",
            "formula": "O2",
            "critical_temperature": 154.58,
            "critical_pressure": 5043000,
            "accentric_factor": 0.0222,
            "radius": 1.51,
            "polar": False,
            "related_column": "outputs.O2-henry_coefficient-mol--kg--Pa_log_cat"
        },
        {
            "name": "nitrogen",
            "formula": "N2",
            "critical_temperature": 126.20,
            "critical_pressure": 3460000,
            "accentric_factor": 0.0377,
            "radius": 1.655,
            "polar": False,
            "related_column": "outputs.N2-henry_coefficient-mol--kg--Pa_log_cat"
        },
    ]
)


# %% ../notebooks/00_data.ipynb 53
def get_opv_data(datadir="../data"):  # path to folder with data files
    """Load the OPV dataset."""
    df = pd.read_csv(os.path.join(datadir, "opv.csv"))
    df = df.groupby(["SMILES", "selfies"]).mean().reset_index()
    return HashableDataFrame(df)

