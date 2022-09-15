import time

import numpy as np
import pandas as pd
from pycm import ConfusionMatrix
from sklearn.model_selection import train_test_split

from gpt3forchem.api_wrappers import (
    extract_regression_prediction,
    fine_tune,
    query_gpt3,
)
from gpt3forchem.baselines import GPRBaseline, compute_fragprints
from gpt3forchem.data import get_photoswitch_data
from gpt3forchem.input import create_single_property_forward_prompts
from gpt3forchem.output import extract_numeric_prediction, get_regression_metrics
import click
from gpt3forchem.helpers import make_if_not_exists
from gpt3forchem.baselines import train_test_gpr_baseline

DF = get_photoswitch_data()

TRAIN_SIZES_NAMES = [10, 40, 60, 70]
TRAIN_SIZES_SMILES = [10, 50, 100, 200, 300, 350]
REPEATS = 10
MODEL_TYPES = ["ada"]
PREFIXES = [""]  # "I'm an expert polymer chemist "]
REPRESENTATIONS = ["SMILES", "selfies", "name"]
OUTDIR = "results/20220915_photoswitch_regression"
make_if_not_exists(OUTDIR)


def learning_curve_point(representation, train_set_size):
    df = DF.dropna(subset=[representation])
    df_train, df_test = train_test_split(
        df, train_size=train_set_size, random_state=None, stratify=df["wavelength_cat"]
    )
    train_prompts = create_single_property_forward_prompts(
        df_train,
        "E isomer pi-pi* wavelength in nm",
        {"E isomer pi-pi* wavelength in nm": "transition wavelength"},
        representation_col=representation,
        encode_value=False,
    )

    test_prompts = create_single_property_forward_prompts(
        df_test,
        "E isomer pi-pi* wavelength in nm",
        {"E isomer pi-pi* wavelength in nm": "transition wavelength"},
        representation_col=representation,
        encode_value=False,
    )

    train_size = len(train_prompts)
    test_size = len(test_prompts)

    filename_base = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    train_filename = f"run_files/{filename_base}_train_prompts_photoswitch_regression_{train_size}.jsonl"
    valid_filename = f"run_files/{filename_base}_valid_prompts_photoswitch_regression_{test_size}.jsonl"

    train_prompts.to_json(train_filename, orient="records", lines=True)
    test_prompts.to_json(valid_filename, orient="records", lines=True)
    train_size = len(train_prompts)
    test_size = len(test_prompts)

    filename_base = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    train_filename = f"run_files/{filename_base}_train_prompts_photoswitch_regression_{train_size}.jsonl"
    valid_filename = f"run_files/{filename_base}_valid_prompts_photoswitch_regression_{test_size}.jsonl"

    train_prompts.to_json(train_filename, orient="records", lines=True)
    test_prompts.to_json(valid_filename, orient="records", lines=True)

    modelname = fine_tune(train_filename, valid_filename, "ada")
    completions = query_gpt3(modelname, test_prompts)
    predictions = [
        extract_regression_prediction(completions, i)
        for i, completion in enumerate(completions["choices"])
    ]
    true = test_prompts["completion"].apply(lambda x: float(x.split("@@")[0]))
    assert len(predictions) == len(true)

    metrics = get_regression_metrics(true, predictions)

    baseline = train_test_gpr_baseline(
        train_filename, valid_filename, representation_column=representation
    )

    baseline_metrics = get_regression_metrics(true, baseline["predictions"])

    results = {
        "train_size": train_size,
        "test_size": test_size,
        "modelname": modelname,
        "representation": representation,
        "completions": completions,
        "predictions": predictions,
        "true": true,
        "train_filename": train_filename,
        "valid_filename": valid_filename,
        "metrics": metrics,
        "baseline_predictions": baseline["predictions"],
        "baseline_metrics": baseline_metrics,
    }

    outname = f"{OUTDIR}/{filename_base}_results_photoswitch_regression_{train_size}_{representation}.pkl"

    save_pickle(outname, results)


if __name__ == "__main__":
    for _ in range(REPEATS):
        for representation in REPRESENTATIONS:
            if representation == "name":
                train_sizes = TRAIN_SIZES_NAMES
            else:
                train_sizes = TRAIN_SIZES_SMILES
            for train_size in train_sizes:
                try:
                    res = learning_curve_point(
                        representation,
                        train_size,
                    )
                    print(
                        f"Finished {representation} {train_size}. R2: {res['metrics']['r2']}, baseline R2: {res['baseline_metrics']['r2']}"
                    )
                    time.sleep(1)
                except Exception as e:
                    print(f"Error: {e}")
                    continue
