from fastcore.helpers import save_pickle
from gpt3forchem.api_wrappers import fine_tune, query_gpt3, extract_regression_prediction
from gpt3forchem.input import create_single_property_forward_prompts_regression
from gpt3forchem.data import get_bandgap_data
import time 
from sklearn.model_selection import train_test_split
from gpt3forchem.helpers import make_if_not_exists
from gpt3forchem.output import get_regression_metrics
import numpy as np 
TRAIN_SIZE = [10, 50, 500]
MAX_TEST_SIZE = 200

REPRESENTATIONS = ['smiles', 'seflies', 'inchi']
OUTDIR = "results/20221206_bandgap_regression"
make_if_not_exists(OUTDIR)

def train_test_bandgap(train_size, representation, random_state=None):
    data = get_bandgap_data()
    train, test = train_test_split(data, train_size=train_size, test_size=MAX_TEST_SIZE, random_state=random_state)
    assert len(test) == MAX_TEST_SIZE
    train_prompts = create_single_property_forward_prompts_regression(train, 'GFN2_HOMO_LUMO_GAP', {'GFN2_HOMO_LUMO_GAP': 'bandgap'}, representation_col=representation, num_digit=2)

    test_prompts = create_single_property_forward_prompts_regression(test, 'GFN2_HOMO_LUMO_GAP', {'GFN2_HOMO_LUMO_GAP': 'bandgap'}, representation_col=representation, num_digit=2)

    filename_base = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    train_filename = f"run_files/{filename_base}_train_prompts_bandgap_regression_{train_size}_{representation}.jsonl"

    valid_filename = f"run_files/{filename_base}_valid_prompts_bandgap_regression_{train_size}_{representation}.jsonl"

    train_prompts.to_json(train_filename, orient="records", lines=True)
    test_prompts.to_json(valid_filename, orient="records", lines=True)

    true = np.array([
            float(test_prompts.iloc[i]["completion"].split("@")[0])
            for i in range(len(test_prompts))
        ])

    modelname = fine_tune(train_filename, valid_filename)

    completions = query_gpt3(modelname, test_prompts)
    predictions = [
            extract_regression_prediction(completions, i)
            for i, completion in enumerate(completions["choices"])
        ]


    result = {
        "train_size": train_size,
        "representation": representation,
        "modelname": modelname,
        "true": true,
        "predictions": predictions,
        "metrics": get_regression_metrics(true, predictions),
        "seed": random_state,
    }

    save_pickle(f"{OUTDIR}/bandgap_{train_size}_{representation}_{random_state}.pkl", result)
    print(f"Train size: {train_size}, Representation: {representation},")

if __name__ == '__main__': 
    for seed in range(1,10):
        for representation in REPRESENTATIONS:
            for train_size in TRAIN_SIZE:
                train_test_bandgap(train_size, representation, seed)

