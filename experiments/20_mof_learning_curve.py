from cProfile import run
import time

from fastcore.utils import save_pickle
from pycm import ConfusionMatrix
from sklearn.model_selection import train_test_split

from gpt3forchem.api_wrappers import extract_prediction, fine_tune, query_gpt3
from gpt3forchem.baselines import XGBClassificationBaseline
from gpt3forchem.data import get_mof_data, discretize
from gpt3forchem.input import create_single_property_forward_prompts

import click

TRAIN_SET_SIZE = [10, 50, 100, 200, 500, 1000, 2000, 3000]
REPEATS = 10
MODEL_TYPES = ["ada"]  # , "ada"]
PREFIXES = [""]  # , "I'm an expert polymer chemist "]

DF = get_mof_data()
RANDOM_STATE = None
MAX_TEST_SIZE = 500  # upper limit to speed it up, this will still require 25 requests

MOFFEATURES = [f for f in DF.columns if f.startswith("features")]


rename_dicts = {
    "outputs.pbe.bandgap_cat": {
        "outputs.pbe.bandgap_cat": "bandgap",
    },
    "outputs.Xe-henry_coefficient-mol--kg--Pa_cat": {
        "outputs.Xe-henry_coefficient-mol--kg--Pa_cat": "Xe Henry coefficient",
    },
    "outputs.Kr-henry_coefficient-mol--kg--Pa_cat": {
        "outputs.Kr-henry_coefficient-mol--kg--Pa_cat": "Kr Henry coefficient",
    },
    "outputs.H2O-henry_coefficient-mol--kg--Pa_cat": {
        "outputs.H2O-henry_coefficient-mol--kg--Pa_cat": "H2O Henry coefficient",
    },
    "outputs.H2S-henry_coefficient-mol--kg--Pa_cat": {
        "outputs.H2S-henry_coefficient-mol--kg--Pa_cat": "H2S Henry coefficient",
    },
    "outputs.CO2-henry_coefficient-mol--kg--Pa_cat": {
        "outputs.CO2-henry_coefficient-mol--kg--Pa_cat": "CO2 Henry coefficient",
    },
    "outputs.CH4-henry_coefficient-mol--kg--Pa_cat": {
        "outputs.CH4-henry_coefficient-mol--kg--Pa_cat": "CH4 Henry coefficient",
    },
    "outputs.O2-henry_coefficient-mol--kg--Pa_cat": {
        "outputs.O2-henry_coefficient-mol--kg--Pa_cat": "O2 Henry coefficient",
    },
}


def learning_curve_point(model_type, train_set_size, prefix, target, representation):
    df = DF.copy()
    discretize(df, target)
    target = f"{target}_cat"
    df = df.dropna(subset=[target, representation])
    df_train, df_test = train_test_split(
        df, train_size=train_set_size, random_state=None, stratify=df[target]
    )
    train_prompts = create_single_property_forward_prompts(
        df_train,
        target,
        rename_dicts[target],
        prompt_prefix=prefix,
        representation_col=representation,  # "info.mofid.mofid_clean",
    )

    test_prompts = create_single_property_forward_prompts(
        df_test,
        target,
        rename_dicts[target],
        prompt_prefix=prefix,
        representation_col=representation,  # "info.mofid.mofid_clean",
    )

    train_size = len(train_prompts)
    test_size = len(test_prompts)

    filename_base = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    train_filename = f"run_files/{filename_base}_train_prompts_mof_{target}_{representation}_{train_size}.jsonl"
    valid_filename = f"run_files/{filename_base}_valid_prompts_mof_{target}_{representation}_{test_size}.jsonl"

    train_prompts.to_json(train_filename, orient="records", lines=True)
    test_prompts.to_json(valid_filename, orient="records", lines=True)

    print(f"Training {model_type} model on {train_size} training examples")
    modelname = fine_tune(train_filename, valid_filename, model_type)
    # taking the first MAX_TEST_SIZE is ok as the train_test_split shuffles the data
    test_prompts = test_prompts.iloc[:MAX_TEST_SIZE]
    completions = query_gpt3(modelname, test_prompts)
    predictions = [
        extract_prediction(completions, i)
        for i, completion in enumerate(completions["choices"][0])
    ]
    true = [
        int(test_prompts.iloc[i]["completion"].split("@")[0])
        for i in range(len(predictions))
    ]
    cm = ConfusionMatrix(true, predictions)

    try:
        baseline = XGBClassificationBaseline(None)
        baseline.tune(df_train[MOFFEATURES], df_train[target])
        baseline.fit(df_train[MOFFEATURES], df_train[target])
        baseline_predictions = baseline.predict(df_test[MOFFEATURES])
        baseline_cm = ConfusionMatrix(df_test[target], baseline_predictions)
        baseline_acc = baseline_cm.ACC_Macro
    except Exception as e:
        print(e)
        baseline_cm = None
        baseline_acc = None

    results = {
        "model_type": model_type,
        "train_set_size": train_set_size,
        "prefix": prefix,
        "train_size": train_size,
        "test_size": test_size,
        "cm": cm,
        "accuracy": cm.ACC_Macro,
        "completions": completions,
        "train_filename": train_filename,
        "valid_filename": valid_filename,
        "MAX_TEST_SIZE": MAX_TEST_SIZE,
        "modelname": modelname,
        "baseline_cm": baseline_cm,
        "baseline_accuracy": baseline_acc,
        "representation": representation,
        "target": target,
    }

    outname = f"results/20220911_mof_classification/{filename_base}_results_mof_{train_size}_{prefix}_{model_type}_{representation}_{target}.pkl"

    save_pickle(outname, results)


@click.command("cli")
@click.argument(
    "target",
    type=click.Choice(
        [
            "outputs.pbe.bandgap",
            "outputs.Xe-henry_coefficient-mol--kg--Pa",
            "outputs.Kr-henry_coefficient-mol--kg--Pa",
            "outputs.H2O-henry_coefficient-mol--kg--Pa",
            "outputs.H2S-henry_coefficient-mol--kg--Pa",
            "outputs.CO2-henry_coefficient-mol--kg--Pa",
            "outputs.CH4-henry_coefficient-mol--kg--Pa",
            "outputs.O2-henry_coefficient-mol--kg--Pa",
        ]
    ),
)
@click.argument(
    "representation", type=click.Choice(["info.mofid.mofid_clean", "chemical_name"])
)
def run_lc(target, representation):
    for _ in range(REPEATS):
        for prefix in PREFIXES:
            for model_type in MODEL_TYPES:
                for train_set_size in TRAIN_SET_SIZE:
                    try:
                        learning_curve_point(
                            model_type, train_set_size, prefix, target, representation
                        )
                    except Exception as e:
                        print(e)


if __name__ == "__main__":
    run_lc()
