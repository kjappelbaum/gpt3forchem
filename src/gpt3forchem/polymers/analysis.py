from dis import dis
from itertools import combinations
import numpy as np
from .featurizer import featurize_many
import pandas as pd
import re
from collections import Counter


def convert2smiles(string):
    new_encoding = {"A": "[Ta]", "B": "[Tr]", "W": "[W]", "R": "[R]"}

    for k, v in new_encoding.items():
        string.replace(k, v)

    string.replace("-", "")

    return string


def get_num_monomer(string, monomer):
    num = re.findall(f"([\d+]) {monomer}", string)
    try:
        num = int(num[0])
    except Exception:
        num = 0
    return num


def get_target(string):
    num = re.findall("([\d+]) adsorption", string)
    return int(num[0])


def get_prompt_compostion(prompt):
    composition = {}

    for monomer in ["R", "W", "A", "B"]:
        composition[monomer] = get_num_monomer(prompt, monomer)

    return composition


def get_prompt_data(prompt):
    composition = get_prompt_compostion(prompt)

    return composition, get_target(prompt)


def get_completion_composition(string):
    parts = string.split("-")
    counts = Counter(parts)
    return dict(counts)


def string2performance(string):
    # we need to perform a bunch of tasks here:
    # 1) Featurize
    # 2) Query the model

    predicted_monomer_sequence = string.split("@")[0].strip()
    monomer_sq = re.findall("[(R|W|A|B)\-(R|W|A|B)]+", predicted_monomer_sequence)[0]
    composition = get_completion_composition(monomer_sq)
    smiles = convert2smiles(predicted_monomer_sequence)
    features = pd.DataFrame(featurize_many([smiles]))

    return {"monomer_squence": monomer_sq, "composition": composition, "smiles": smiles}


def composition_mismatch(composition: dict, found: dict):
    distances = []

    # We also might have the case the there are keys that the input did not contain
    all_keys = set(combinations.keys()) & set(found.keys())

    expected_len = []
    found_len = []

    for key in all_keys:
        try:
            expected = combinations[key]
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
