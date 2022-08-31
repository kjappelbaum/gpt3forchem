# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/04_output.ipynb.

# %% auto 0
__all__ = ['string_distances', 'convert2smiles', 'get_num_monomer', 'get_prompt_compostion', 'get_target', 'get_prompt_data',
           'get_completion_composition', 'string2performance', 'composition_mismatch', 'get_regression_metrics']

# %% ../notebooks/04_output.ipynb 1
import re
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
from nbdev.showdoc import *
from strsimpy.levenshtein import Levenshtein
from strsimpy.longest_common_subsequence import LongestCommonSubsequence
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
import pandas as pd

from sklearn.metrics import r2_score, max_error, mean_absolute_error, mean_squared_error


# %% ../notebooks/04_output.ipynb 4
def string_distances(
    training_set: Iterable[str], # string representations of the compounds in the training set
    query_string: str # string representation of the compound to be queried
):

    distances = defaultdict(list)

    metrics = [
        ("Levenshtein", Levenshtein()),
        ("NormalizedLevenshtein", NormalizedLevenshtein()),
        ("LongestCommonSubsequence", LongestCommonSubsequence()),
    ]

    aggregations = [
        ("min", lambda x: np.min(x)),
        ("max", lambda x: np.max(x)),
        ("mean", lambda x: np.mean(x)),
        ("std", lambda x: np.std(x)),
    ]

    for training_string in training_set:
        for metric_name, metric in metrics:
            distances[metric_name].append(
                metric.distance(training_string, query_string)
            )

    aggregated_distances = {}

    for k, v in distances.items():
        for agg_name, agg_func in aggregations:
            aggregated_distances[f"{k}_{agg_name}"] = agg_func(v)

    return aggregated_distances


# %% ../notebooks/04_output.ipynb 7
def convert2smiles(string):
    new_encoding = {"A": "[Ta]", "B": "[Tr]", "W": "[W]", "R": "[R]"}

    for k, v in new_encoding.items():
        string = string.replace(k, v)

    string = string.replace("-", "")

    return string


# %% ../notebooks/04_output.ipynb 11
def get_num_monomer(string, monomer):
    num = re.findall(f"([\d+]) {monomer}", string)
    try:
        num = int(num[0])
    except Exception:
        num = 0
    return num


# %% ../notebooks/04_output.ipynb 13
def get_prompt_compostion(prompt):
    composition = {}

    for monomer in ["R", "W", "A", "B"]:
        composition[monomer] = get_num_monomer(prompt, monomer)

    return composition


# %% ../notebooks/04_output.ipynb 14
def get_target(string, target_name="adsorption"):
    num = re.findall(f"([\d+]) {target_name}", string)
    return int(num[0])


# %% ../notebooks/04_output.ipynb 15
def get_prompt_data(prompt):
    composition = get_prompt_compostion(prompt)

    return composition, get_target(prompt)


# %% ../notebooks/04_output.ipynb 16
def get_completion_composition(string):
    parts = string.split("-")
    counts = Counter(parts)
    return dict(counts)


# %% ../notebooks/04_output.ipynb 17
def string2performance(string):
    # we need to perform a bunch of tasks here:
    # 1) Featurize
    # 2) Query the model

    predicted_monomer_sequence = string.split("@")[0].strip()
    monomer_sq = re.findall("[(R|W|A|B)\-(R|W|A|B)]+", predicted_monomer_sequence)[0]
    composition = get_completion_composition(monomer_sq)
    smiles = convert2smiles(predicted_monomer_sequence)

    features = pd.DataFrame(featurize_many([smiles]))
    prediction = DELTA_G_MODEL.predict(features[FEATURES])
    return {
        "monomer_squence": monomer_sq,
        "composition": composition,
        "smiles": smiles,
        "prediction": prediction,
    }


# %% ../notebooks/04_output.ipynb 18
def composition_mismatch(composition: dict, found: dict):
    distances = []

    # We also might have the case the there are keys that the input did not contain
    all_keys = set(composition.keys()) & set(found.keys())

    expected_len = []
    found_len = []

    for key in all_keys:
        try:
            expected = composition[key]
        except KeyError:
            expected = 0
        expected_len.append(expected)
        try:
            f = found[key]
        except KeyError:
            f = 0
        found_len.append(f)

        distances.append(np.abs(expected - f))

    expected_len = sum(expected_len)
    found_len = sum(found_len)
    return {
        "distances": distances,
        "min": np.min(distances),
        "max": np.max(distances),
        "mean": np.mean(distances),
        "expected_len": expected_len,
        "found_len": found_len,
    }


# %% ../notebooks/04_output.ipynb 19
def get_regression_metrics(
    y_true,  # actual values (ArrayLike)
    y_pred,  # predicted values (ArrayLike)
) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
        "mean_absolute_error": mean_absolute_error(y_true, y_pred),
        "mean_squared_error": mean_squared_error(y_true, y_pred),
    }

