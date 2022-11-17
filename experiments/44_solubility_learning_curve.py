import time

from fastcore.utils import save_pickle
from sklearn.model_selection import train_test_split

from gpt3forchem.api_wrappers import extract_prediction, fine_tune, query_gpt3, extract_regression_prediction
from gpt3forchem.data import get_solubility_data
from gpt3forchem.input import create_prompts_solubility, _SOLUBILITY_FEATURES, encode_categorical_value
from gpt3forchem.output import get_regression_metrics
from pycm import ConfusionMatrix
import numpy as np 

from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def run_tabpfn_baseline(train_data, test_data):
    X_train, y_train = train_data[_SOLUBILITY_FEATURES], train_data['Solubility_cat']
    X_test, y_test = test_data[_SOLUBILITY_FEATURES], test_data['Solubility_cat']
    classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)

    classifier.fit(X_train, y_train)
    y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)
    cm = ConfusionMatrix(y_test.values, y_eval)

    return cm, y_eval


def run_xgboost_baseline(train_data, test_data):
    X_train, y_train = train_data[_SOLUBILITY_FEATURES], train_data['Solubility_cat'].map(encode_categorical_value)
    X_test, y_test = test_data[_SOLUBILITY_FEATURES], test_data['Solubility_cat'].map(encode_categorical_value) 
    classifier = XGBClassifier(n_estimators=5000)

    classifier.fit(X_train, y_train)
    y_eval = classifier.predict(X_test)
    cm = ConfusionMatrix(y_test.values, y_eval)

    return cm, y_eval


def train_test_gpts(train_data, test_data, repr, regression=False, subsample:int=50): 
    train_prompts = create_prompts_solubility(train_data, representation=repr, regression=regression)
    test_prompts = create_prompts_solubility(test_data, representation=repr, regression=regression)

    train_size  = len(train_prompts)
    test_size = len(test_prompts)

    filename_base = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    train_filename = f"run_files/{filename_base}_train_prompts_solubility_classification_{train_size}.jsonl"
    valid_filename = f"run_files/{filename_base}_valid_prompts_solubility_classification_{test_size}.jsonl"

    train_prompts.to_json(train_filename, orient="records", lines=True)
    test_prompts.to_json(valid_filename, orient="records", lines=True)

    modelname = fine_tune(train_filename, valid_filename)

    completions = query_gpt3(modelname, test_prompts) 
    predictions = [extract_prediction(completions, i) for i, completion in enumerate(completions["choices"])]
    true = [test_prompts.iloc[i]['completion'].split('@')[0] for i in range(len(test_prompts))]
    return modelname, predictions, completions, true, ConfusionMatrix(true, predictions)


MAX_TEST_SIZE = 500
TRAIN_SIZES = [10, 50, 500]
OUTDIR = 'results/20221117_solubility'
REPRESENTATIONS = ['SMILES', 'InChI', 'selfies', 'iupac_names', 'Name']
REPEATS = 10

def learning_curve_point(num_train_points, outdir, representation='smiles', random_state=42):
    data = get_solubility_data()
    train_data, test_data = train_test_split(data, train_size=num_train_points, test_size=MAX_TEST_SIZE, random_state=random_state, stratify=data['Solubility_cat'])

    baseline_cm, baseline_y_eval = run_tabpfn_baseline(train_data, test_data)
    modelname, predictions, completions, true, cm = train_test_gpts(train_data, test_data, representation, regression=False)

    res = {
        'num_train_points': num_train_points,
        'baseline_cm': baseline_cm,
        'baseline_y_eval': baseline_y_eval,
        'modelname': modelname,
        'predictions': predictions,
        'completions': completions,
        'true': true,
        'cm': cm,
    }
    
    outname = f"{outdir}/results_solubility_{num_train_points}.pkl"

    save_pickle(outname, res)

if __name__ == '__main__':
    for _ in range(REPEATS):
        random_seed = np.random.randint(0, 100000)
        for train_size in TRAIN_SIZES:
            for representation in REPRESENTATIONS:
                learning_curve_point(train_size, OUTDIR, representation=representation, random_state=random_seed)