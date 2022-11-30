from gpt3forchem.data import get_lipophilicity_data
from gpt3forchem.api_wrappers import fine_tune, query_gpt3, extract_prediction
import time
from fastcore.utils import save_pickle
from gpt3forchem.input import create_single_property_forward_prompts
from gpt3forchem.baselines import train_test_gauche, BEST_GAUCHE_MODEL_CONFIG
from sklearn.model_selection import train_test_split
from gpt3forchem.helpers import make_if_not_exists
from gpt3forchem.output import get_regression_metrics

TRAIN_SIZES = [10, 50, 500]
N_REPEATS = 10
MAX_TEST_SIZE = 200
REPRESENTATIONS = ['smiles', 'selfies', 'inchi', 'iupac_name']

OUTDIR = "results/20221129_lipophilicity_regression"
make_if_not_exists(OUTDIR)

def train_test_freesolv(train_size, representation, random_state=None):
    df = get_lipophilicity_data()
    test_size = min(MAX_TEST_SIZE, len(df)-train_size)
    df['exp'] = df['exp'].round(2)
    train, test = train_test_split(df, train_size=train_size, test_size=test_size,  random_state=random_state, stratify=df['exp_cat'])
    model_config = BEST_GAUCHE_MODEL_CONFIG['lipophilicity']
    baseline = train_test_gauche(train_size, test_size, 'Lipophilicity', '../data/lipophilicity.csv', model_config['featurizer'], model_config['model'], regression=True, random_state=random_state)

    train_prompts = create_single_property_forward_prompts(train, 'exp', {'exp': 'lipophilicity'}, representation_col=representation, encode_value=False)
    test_prompts = create_single_property_forward_prompts(test, 'exp', {'exp': 'lipophilicity'}, representation_col=representation, encode_value=False)

    filename_base = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    train_filename = f"run_files/{filename_base}_train_prompts_lipophilicity_regression_{representation}_{train_size}.jsonl"
    valid_filename = f"run_files/{filename_base}_valid_prompts_lipophilicity_regression_{representation}_{test_size}.jsonl"

    train_prompts.to_json(train_filename, orient="records", lines=True)
    test_prompts.to_json(valid_filename, orient="records", lines=True)

    modelname = fine_tune(train_filename, valid_filename)

    completions = query_gpt3(modelname, test_prompts)
    predictions = [
            float(extract_prediction(completions, i))
            for i, completion in enumerate(completions["choices"])
        ]
    true =[
            float(test_prompts.iloc[i]["completion"].split("@")[0])
            for i in range(len(test_prompts))
        ]
    assert len(predictions) == len(true[:MAX_TEST_SIZE])

    metrics = get_regression_metrics(true[:MAX_TEST_SIZE], predictions)
    outputs = {
        'baseline': baseline,
        'train_size': train_size,
        'representation': representation,
        'modelname': modelname,
        'train_filename': train_filename,
        'valid_filename': valid_filename,
        'completions': completions,
        'metrics': metrics
    }
    print(f"Train size: {train_size}, representation {representation} | MAE: {metrics['mean_absolute_error']}, Baseline MAE: {baseline['mean_absolute_error']}")
    save_pickle(f"{OUTDIR}/{filename_base}_lipophilicity_regression_{representation}_{train_size}.pkl", outputs)   


if __name__ == "__main__":
    for i in range(N_REPEATS):
        for representation in REPRESENTATIONS:
            for train_size in TRAIN_SIZES:
                train_test_freesolv(train_size, representation, random_state=i)