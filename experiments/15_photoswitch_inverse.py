import time

from fastcore.xtras import save_pickle
from sklearn.model_selection import train_test_split

from gpt3forchem.api_wrappers import fine_tune
from gpt3forchem.data import get_photoswitch_data
from gpt3forchem.input import generate_inverse_photoswitch_prompts
from gpt3forchem.output import test_inverse_photoswitch
from gpt3forchem.helpers import make_if_not_exists

REPEATS = 10
TRAIN_TEST_RATIO = 0.8

# SMILES was the most predictive representation, names are not invertible.
# Hence, we will use SMILES
DF = get_photoswitch_data()
TRAIN_TEST_RATIO = (
    0.9  # we hold out some data to get some "fresh" but still "reasonable" queries
)

TEMPERATURES = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
MODEL_TYPE = "ada"
N_EPOCHS = 4
OUTDIR = "results/20220915_inverse_photoswitch"
make_if_not_exists(OUTDIR)


def inverse_run():
    prompts = generate_inverse_photoswitch_prompts(DF)
    train_prompts, test_prompts = train_test_split(prompts, train_size=TRAIN_TEST_RATIO)

    train_size = len(train_prompts)
    test_size = len(test_prompts)

    filename_base = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    train_filename = f"run_files/{filename_base}_train_prompts_photoswitch_inverse_{train_size}.jsonl"
    valid_filename = (
        f"run_files/{filename_base}_valid_prompts_photoswitch_inverse_{test_size}.jsonl"
    )

    train_prompts.to_json(train_filename, orient="records", lines=True)
    test_prompts.to_json(valid_filename, orient="records", lines=True)

    modelname = fine_tune(train_filename, valid_filename, MODEL_TYPE, n_epochs=N_EPOCHS)

    train_smiles = (
        train_prompts["completion"].apply(lambda x: x.replace("@@@", "")).to_list()
    )

    results = []
    for temperature in TEMPERATURES:
        result = test_inverse_photoswitch(
            test_prompts, modelname, train_smiles=train_smiles, temperature=temperature
        )

        result["train_filename"] = train_filename
        result["valid_filename"] = valid_filename
        results.append(result)

    save_pickle(f"{OUTDIR}/{filename_base}_metrics.pkl", results)


if __name__ == "__main__":
    for _ in range(REPEATS):
        inverse_run()
