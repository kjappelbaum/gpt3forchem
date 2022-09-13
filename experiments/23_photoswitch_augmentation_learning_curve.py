import time

from fastcore.xtras import save_pickle
from pycm import ConfusionMatrix
from sklearn.model_selection import train_test_split

from gpt3forchem.api_wrappers import extract_prediction, fine_tune, query_gpt3
from gpt3forchem.baselines import train_test_gpr_baseline
from gpt3forchem.data import get_photoswitch_data
from gpt3forchem.input import create_single_property_forward_prompts

TRAIN_SIZES_NAMES = [10, 40, 60, 70]
TRAIN_SIZES_SMILES = [10, 50, 100, 200, 300, 350]

REPRESENTATIONS = ["SMILES", "selfies", "name"]

REPEATS = 10
DF = get_photoswitch_data()
MODEL_TYPE = "ada"
PREFIX = ""
N_EPOCHS = 4  # this is the default


def learning_curve_point(representation, model_type, train_set_size):
    df = DF.dropna(subset=[representation])
    df_train, df_test = train_test_split(
        df, train_size=train_set_size, random_state=None, stratify=df["wavelength_cat"]
    )
    train_prompts = create_single_property_forward_prompts(
        df_train,
        "wavelength_cat",
        {"wavelength_cat": "transition wavelength"},
        representation_col=representation,
        smiles_augmentation=True,
    )

    test_prompts = create_single_property_forward_prompts(
        df_test,
        "wavelength_cat",
        {"wavelength_cat": "transition wavelength"},
        representation_col=representation,
        smiles_augmentation=True,
    )

    train_size = len(train_prompts)
    test_size = len(test_prompts)

    filename_base = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    train_filename = f"run_files/{filename_base}_train_prompts_photoswitch_{train_size}_{representation}.jsonl"
    valid_filename = f"run_files/{filename_base}_valid_prompts_photoswitch_{test_size}_{representation}.jsonl"

    train_prompts.to_json(train_filename, orient="records", lines=True)
    test_prompts.to_json(valid_filename, orient="records", lines=True)

    print(f"Training {model_type} model on {train_size} training examples")
    modelname = fine_tune(train_filename, valid_filename, model_type, n_epochs=N_EPOCHS)

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

    baseline = train_test_gpr_baseline(
        train_filename, valid_filename, representation_column=representation
    )
    results = {
        "model_type": model_type,
        "train_set_size": train_set_size,
        # "prefix": prefix,
        "train_size": train_size,
        "test_size": test_size,
        "cm": cm,
        "accuracy": cm.ACC_Macro,
        "completions": completions,
        "train_filename": train_filename,
        "valid_filename": valid_filename,
        "modelname": modelname,
        "baseline": baseline,
        "representation": representation,
        "baseline_accuracy": baseline["cm"].ACC_Macro,
    }

    outname = f"results/photoswitch_{N_EPOCHS}epoch/{filename_base}_results_photoswitch_{train_size}_{model_type}_{representation}.pkl"

    save_pickle(outname, results)
    return results


if __name__ == "__main__":
    for _ in range(REPEATS):
        for representation in REPRESENTATIONS:
            if representation == "name":
                train_sizes = TRAIN_SIZES_NAMES
            else:
                train_sizes = TRAIN_SIZES_SMILES
            for train_size in train_sizes:
                res = learning_curve_point(representation, MODEL_TYPE, train_size)
                print(
                    f"Finished {representation} {train_size}. Accuracy: {res['accuracy']}. Baseline accuracy: {res['baseline_accuracy']}"
                )
                time.sleep(1)
