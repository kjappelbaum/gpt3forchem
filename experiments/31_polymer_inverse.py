from fastcore.helpers import save_pickle
from sklearn.model_selection import train_test_split

from gpt3forchem.api_wrappers import extract_prediction, fine_tune, query_gpt3
from gpt3forchem.data import get_polymer_data
from gpt3forchem.helpers import get_bin_ranges, make_if_not_exists
from gpt3forchem.output import get_inverse_polymer_metrics
from gpt3forchem.input import create_single_property_inverse_polymer_prompts
import time
import click

DF = get_polymer_data()
bins = get_bin_ranges(DF, "deltaGmin", 5)
OUTDIR = "results/20221117_polymer_inverse"
make_if_not_exists(OUTDIR)
TEMPERATURES = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def get_inverse_point(exclude_category=None):
    print(f"Excluding category {exclude_category}")
    if exclude_category is None:
        train_df, test_df = train_test_split(
            DF, train_size=0.9, random_state=None, stratify=DF["deltaGmin_cat"], 
        )
    else: 
        train_df = DF[~DF["deltaGmin_cat"].isin([exclude_category])]
        test_df = DF[DF["deltaGmin_cat"].isin([exclude_category])]
    train_prompts = create_single_property_inverse_polymer_prompts(
        train_df,
        "deltaGmin_cat",
        {"deltaGmin_cat": "adsorption energy"},
        encode_value=False,
    )

    test_prompts = create_single_property_inverse_polymer_prompts(
        test_df,
        "deltaGmin_cat",
        {"deltaGmin_cat": "adsorption energy"},
        encode_value=False,
    )

    train_size = len(train_prompts)
    test_size = len(test_prompts)

    filename_base = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    train_filename = (
        f"run_files/{filename_base}_train_prompts_polymer_inverse_{train_size}.jsonl"
    )
    valid_filename = (
        f"run_files/{filename_base}_valid_prompts_polymer_inverse_{test_size}.jsonl"
    )

    train_prompts.to_json(train_filename, orient="records", lines=True)
    test_prompts.to_json(valid_filename, orient="records", lines=True)

    modelname = fine_tune(train_filename, valid_filename, "ada")

    t_dependend_metrics = {}
    t_dependend_predictions = {}
    t_dependend_completions = {}
    for temperature in TEMPERATURES:
        completions = query_gpt3(
            modelname, test_prompts, max_tokens=200, temperature=temperature
        )
        predictions = [
            extract_prediction(completions, i)
            for i, completion in enumerate(completions["choices"])
        ]

        prediction_metrics = get_inverse_polymer_metrics(
            predictions, test_prompts, train_prompts, bins, max_num_train_sequences=2000
        )
        t_dependend_metrics[temperature] = prediction_metrics
        t_dependend_predictions[temperature] = predictions
        t_dependend_completions[temperature] = completions

    true = [prompt.split("@")[0] for prompt in test_prompts["completion"]]

    optimal_metrics = get_inverse_polymer_metrics(
        true, test_prompts, train_prompts, bins, max_num_train_sequences=2000
    )

    results = {
        "train_file": train_filename,
        "valid_file": valid_filename,
        "modelname": modelname,
        "predictions": t_dependend_predictions,
        "completions": t_dependend_completions,
        "metrics": t_dependend_metrics,
        "optimal_metrics": optimal_metrics,
        "num_train_points": train_size,
        "num_test_points": test_size,
        "exclude_category": exclude_category,
    }

    save_pickle(f"{OUTDIR}/{filename_base}_results.pkl", results)


@click.command()
@click.option("--exclude_category", default=None, type=str)
def main(exclude_category):
    for _ in range(10):
        get_inverse_point(exclude_category=exclude_category)

if __name__ == "__main__":
    main()
