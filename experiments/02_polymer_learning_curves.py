import time

from fastcore.utils import save_pickle
from pycm import ConfusionMatrix
from sklearn.model_selection import train_test_split

from gpt3forchem.api_wrappers import extract_prediction, fine_tune, query_gpt3
from gpt3forchem.data import get_polymer_data
from gpt3forchem.input import create_single_property_forward_prompts

TRAIN_SET_SIZE = [10, 50, 100, 200, 500, 1000, 2000, 3000]
REPEATS = 10
MODEL_TYPES = ["ada"]  # , "ada"]
PREFIXES = [""]  # , "I'm an expert polymer chemist "]

DF = get_polymer_data()
RANDOM_STATE = None
MAX_TEST_SIZE = 500  # upper limit to speed it up, this will still require 25 requests


def learning_curve_point(model_type, train_set_size, prefix):
    df_train, df_test = train_test_split(
        DF, train_size=train_set_size, random_state=None
    )
    train_prompts = create_single_property_forward_prompts(
        df_train,
        "deltaGmin_cat",
        {"deltaGmin_cat": "adsorption energy"},
        prompt_prefix=prefix,
    )

    test_prompts = create_single_property_forward_prompts(
        df_test,
        "deltaGmin_cat",
        {"deltaGmin_cat": "adsorption energy"},
        prompt_prefix=prefix,
    )

    train_size = len(train_prompts)
    test_size = len(test_prompts)

    filename_base = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    train_filename = (
        f"run_files/{filename_base}_train_prompts_polymers_{train_size}.jsonl"
    )
    valid_filename = (
        f"run_files/{filename_base}_valid_prompts_polymers_{test_size}.jsonl"
    )

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
    }

    outname = f"results/{filename_base}_results_polymers_{train_size}_{prefix}_{model_type}.pkl"

    save_pickle(outname, results)


if __name__ == "__main__":
    for repeat in range(REPEATS):
        for model_type in MODEL_TYPES:
            for train_set_size in TRAIN_SET_SIZE:
                for prefix in PREFIXES:
                    try:
                        learning_curve_point(model_type, train_set_size, prefix)
                    except Exception as e:
                        time.sleep(10)
                        print(e)