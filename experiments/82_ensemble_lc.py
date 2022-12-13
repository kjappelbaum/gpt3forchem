import pandas as pd 
from gpt3forchem.api_wrappers import fine_tune, query_gpt3, extract_prediction
from gpt3forchem.data import get_waterstability_data
from sklearn.model_selection import train_test_split
from pycm import ConfusionMatrix
import time

import numpy as np

from scipy.stats import mode
from fastcore.xtras import save_pickle


PROMPT_TEMPLATE_water_stability= "How is the water stability of {}"###"
COMPLETION_TEMPLATE_water_stability = "{}@@@"


OUTDIR = "results/20221213_waterstability_inverse"

def generate_water_stability_prompts(
    data: pd.DataFrame
) -> pd.DataFrame:
    prompts = []
    completions = []
    for i, row in data.iterrows():

        prompt = PROMPT_TEMPLATE_water_stability.format(
            row['normalized_names']
        )

        stability = 0 if row['stability'] == 'low' else 1
        completion = COMPLETION_TEMPLATE_water_stability.format(stability)
        prompts.append(prompt)
        completions.append(completion)

    prompts = pd.DataFrame(
        {"prompt": prompts, "completion": completions,}
    )

    return prompts


def train_test_waterstability(train_size, random_state=None, num_models=5):
    data = get_waterstability_data()
    train_data, test_data = train_test_split(data, train_size=train_size, random_state=random_state, stratify=data['stability'])
    true = test_data['stability'].apply(lambda x: 0 if x == 'low' else 1.).values

    models = []

    for i in range(num_models):
        # resample the training set with replacement
        train_data_resampled = train_data.sample(n=len(train_data), replace=True)
        prompts = generate_water_stability_prompts(train_data_resampled)
        filename_base = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        train_filename = f"run_files/{filename_base}_train_prompts_water_stability.jsonl"
        valid_filename = f"run_files/{filename_base}_valid_prompts_water_stability.jsonl"
        test_prompts = generate_water_stability_prompts(test_data)

        prompts.to_json(train_filename, orient="records", lines=True)
        test_prompts.to_json(valid_filename, orient="records", lines=True)

        model = fine_tune(train_filename, valid_filename)
        models.append(model)

    overall_predictions = []
    overall_completions = []

    for model in models:
        completions = query_gpt3(model, test_prompts)
        predictions = []

        for i in range(len(completions['choices'])):
            try:
                pred = int(extract_prediction(completions, i))
            except:
                pred = np.nan
            predictions.append(pred)
        overall_predictions.append(predictions)
        overall_completions.append(completions)

    true = test_data['stability'].apply(lambda x: 0 if x == 'low' else 1.).values

    overall_predictions = np.array(overall_predictions)
    mode_pred = mode(overall_predictions, axis=0).mode.flatten()
    std = np.std(overall_predictions, axis=0)

    high_conf_mask = std ==0 
    high_conf_pred= mode_pred[high_conf_mask]
    high_conf_true = true[high_conf_mask]
    high_conf_cm = ConfusionMatrix(high_conf_true, high_conf_pred)

    res = {
        'true': true,
        'mode': mode_pred,
        'std': std,
        "seed": random_state,
        "models": models,
        'predictions': overall_predictions,
        'completions': overall_completions,
        "mode_cm": ConfusionMatrix(true, mode_pred),
        "high_conf_cm": high_conf_cm,
    }
    save_pickle(f"{OUTDIR}/{filename_base}_waterstability_ensemble_{train_size}_{random_state}_{num_models}.pkl", res)

if __name__ == '__main__':
    for i in range(10):
        for num_models in [5, 10, 50]:
            train_test_waterstability(train_size=0.8, random_state=i, num_models=num_models)