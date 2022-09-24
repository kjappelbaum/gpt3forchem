import time

from fastcore.utils import save_pickle
from pycm import ConfusionMatrix
from sklearn.model_selection import train_test_split

from gpt3forchem.api_wrappers import (
    extract_regression_prediction,
    fine_tune,
    query_gpt3,
)
from gpt3forchem.baselines import XGBRegressionBaseline
from gpt3forchem.data import get_mof_data, discretize
from gpt3forchem.input import create_single_property_forward_prompts
from gpt3forchem.helpers import make_if_not_exists
from gpt3forchem.output import get_regression_metrics
import click
import numpy as np

TRAIN_SET_SIZE = [10, 50, 100, 200, 500, 1000, 2000, 3000]
REPEATS = 10
MODEL_TYPES = ["ada"]  # , "ada"]
PREFIXES = [""]  # , "I'm an expert polymer chemist "]

DF = get_mof_data()
RANDOM_STATE = None
MAX_TEST_SIZE = 500  # upper limit to speed it up, this will still require 25 requests

MOFFEATURES = [f for f in DF.columns if f.startswith("features")]
OUTDIR = "results/20220915_mof_regression"
make_if_not_exists(OUTDIR)
rename_dicts = {
    "outputs.pbe.bandgap": {
        "outputs.pbe.bandgap": "bandgap",
    },
    "outputs.Xe-henry_coefficient-mol--kg--Pa_log": {
        "outputs.Xe-henry_coefficient-mol--kg--Pa_cat": "Xe Henry coefficient",
    },
    "outputs.Kr-henry_coefficient-mol--kg--Pa_log": {
        "outputs.Kr-henry_coefficient-mol--kg--Pa_cat": "Kr Henry coefficient",
    },
    "outputs.H2O-henry_coefficient-mol--kg--Pa_log": {
        "outputs.H2O-henry_coefficient-mol--kg--Pa_log": "H2O Henry coefficient",
    },
    "outputs.H2S-henry_coefficient-mol--kg--Pa_log": {
        "outputs.H2S-henry_coefficient-mol--kg--Pa_log": "H2S Henry coefficient",
    },
    "outputs.CO2-henry_coefficient-mol--kg--Pa_log": {
        "outputs.CO2-henry_coefficient-mol--kg--Pa_log": "CO2 Henry coefficient",
    },
    "outputs.CH4-henry_coefficient-mol--kg--Pa_log": {
        "outputs.CH4-henry_coefficient-mol--kg--Pa_log": "CH4 Henry coefficient",
    },
    "outputs.O2-henry_coefficient-mol--kg--Pa_log": {
        "outputs.O2-henry_coefficient-mol--kg--Pa_log": "O2 Henry coefficient",
    },
}


def learning_curve_point(
    model_type, train_set_size, prefix, target, representation, only_baseline
):
    df = DF.copy()
    df = df.dropna(subset=[target, representation])
    df_train, df_test = train_test_split(
        df, train_size=train_set_size, random_state=None
    )
    train_prompts = create_single_property_forward_prompts(
        df_train,
        target,
        rename_dicts[target],
        representation_col=representation,  # "info.mofid.mofid_clean",
        encode_value=False,
    )

    test_prompts = create_single_property_forward_prompts(
        df_test,
        target,
        rename_dicts[target],
        representation_col=representation,  # "info.mofid.mofid_clean",
        encode_value=False,
    )

    train_size = len(train_prompts)
    test_size = len(test_prompts)

    filename_base = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    train_filename = f"run_files/{filename_base}_train_prompts_mof_regression_{target}_{representation}_{train_size}.jsonl"
    valid_filename = f"run_files/{filename_base}_valid_prompts_mof_regression_{target}_{representation}_{test_size}.jsonl"

    train_prompts.to_json(train_filename, orient="records", lines=True)
    test_prompts.to_json(valid_filename, orient="records", lines=True)

    print(f"Training {model_type} model on {train_size} training examples")
    true = [
            float(test_prompts.iloc[i]["completion"].split("@")[0])
            for i in range(len(test_prompts))
        ]
    if not only_baseline:
        modelname = fine_tune(train_filename, valid_filename, model_type)

        # taking the first MAX_TEST_SIZE is ok as the train_test_split shuffles the data
        test_prompts = test_prompts.iloc[:MAX_TEST_SIZE]
        completions = query_gpt3(modelname, test_prompts)
        predictions = [
            extract_regression_prediction(completions, i)
            for i, completion in enumerate(completions["choices"])
        ]

        assert len(predictions) == len(true)
        metrics = get_regression_metrics(true, predictions)
    else:
        metrics = None
        predictions = None
        completions = None
        modelname = None

    baseline = XGBRegressionBaseline(None)
    baseline.tune(df_train[MOFFEATURES], df_train[target])
    baseline.fit(df_train[MOFFEATURES], df_train[target])
    baseline_predictions = baseline.predict(df_test[MOFFEATURES])

    baseline_metrics = get_regression_metrics(true, baseline_predictions)

    results = {
        "model_type": model_type,
        "train_set_size": train_set_size,
        "prefix": prefix,
        "train_size": train_size,
        "test_size": test_size,
        "metrics": metrics,
        "completions": completions,
        "train_filename": train_filename,
        "valid_filename": valid_filename,
        "MAX_TEST_SIZE": MAX_TEST_SIZE,
        "modelname": modelname,
        "baseline_metrics": baseline_metrics,
        "representation": representation,
        "target": target,
    }

    outname = f"{OUTDIR}/{filename_base}_results_mof_regression_{train_size}_{prefix}_{model_type}_{representation}_{target}.pkl"

    save_pickle(outname, results)


@click.command("cli")
@click.argument(
    "target",
    type=click.Choice(
        [
            "outputs.pbe.bandgap",
            "outputs.Xe-henry_coefficient-mol--kg--Pa_log",
            "outputs.Kr-henry_coefficient-mol--kg--Pa_log",
            "outputs.H2O-henry_coefficient-mol--kg--Pa_log",
            "outputs.H2S-henry_coefficient-mol--kg--Pa_log",
            "outputs.CO2-henry_coefficient-mol--kg--Pa_log",
            "outputs.CH4-henry_coefficient-mol--kg--Pa_log",
            "outputs.O2-henry_coefficient-mol--kg--Pa_log",
        ]
    ),
)
@click.argument(
    "representation", type=click.Choice(["info.mofid.mofid_clean", "chemical_name"])
)
@click.option("--only_baseline", is_flag=True)
def run_lc(target, representation, only_baseline):
    for _ in range(REPEATS):
        for prefix in PREFIXES:
            for model_type in MODEL_TYPES:
                for train_set_size in TRAIN_SET_SIZE[::-1]:

                    learning_curve_point(
                        model_type,
                        train_set_size,
                        prefix,
                        target,
                        representation,
                        only_baseline,
                    )
      

if __name__ == "__main__":
    run_lc()
