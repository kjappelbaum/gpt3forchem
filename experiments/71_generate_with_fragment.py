from gpt3forchem.input import generate_inverse_photoswitch_prompts_with_fragment
from gpt3forchem.data import get_bandgap_data
from gpt3forchem.output import test_inverse_bandgap
import pandas as pd
from fastcore.helpers import save_pickle
from scipy import stats
from gpt3forchem.helpers import make_if_not_exists
import time 

FRAGMENTS = ["I", "Cl", "F", "C#CBr", "C#CC", "C(=O)", "C#C", "Br"] #C(=O)", "C#C", "Br", 
TEMPERATURES = [0, 0.05, 0.15, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2][::-1]
TRAIN_PROMPTS = pd.read_csv("run_files/2022-11-27-22-29-43500000_smiles_train.csv")
OUTDIR = "results/20221206_generate_with_fragment"
NUM_POINTS = 2000
make_if_not_exists(OUTDIR)

def test_inverse_model(
    modelname,
    test_prompts,
    df_train,
    max_tokens: int = 250,
    temperatures=None,
    representation="SMILES",
):
    temperatures = temperatures or [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    train_smiles = df_train["SMILES"].to_list()
    results = []
    for temperature in temperatures:
        try:
            print(f"Testing temperature {temperature} for {representation}")
            result = test_inverse_bandgap(
                test_prompts,
                modelname,
                train_smiles=train_smiles,
                temperature=temperature,
                max_tokens=max_tokens,
                representation=representation,
            )
            results.append(result)
        except Exception as e:
            print(e)
            pass

    return results


def prevalence(df, fragment):
    return df["smiles"].apply(lambda x: fragment in x).sum() / len(df)


def train_test(fragment, num_points, temperatures):
    data = get_bandgap_data()
    test_data = data.sample(num_points)
    test_data["fragment"] = fragment
    prevalence_of_fragment = prevalence(data, fragment)
    prompts = generate_inverse_photoswitch_prompts_with_fragment(test_data)

    test_2_res = test_inverse_model(
        "ada:ft-lsmoepfl-2022-11-28-16-40-07",
        prompts,
        TRAIN_PROMPTS,
        representation="SMILES",
        temperatures=temperatures,
    )

    test2_smiles = []

    for res in test_2_res:
        result = {}
        result["temperature"] = res["meta"]["temperature"]
        result["smiles"] = set(res["predictions"][res["valid_smiles"]])
        fragment_in_smiles = [fragment in s for s in result["smiles"]]
        result["fragment_in_smiles"] = (
            sum(fragment_in_smiles) / len(result["smiles"])
            if len(result["smiles"]) > 0
            else 0
        )

        expected_pos = int(prevalence_of_fragment * len(result["smiles"]))
        expected = [expected_pos, len(result["smiles"]) - expected_pos]
        found = [
            sum(fragment_in_smiles),
            len(result["smiles"]) - sum(fragment_in_smiles),
        ]
        p_value = stats.chisquare(found, expected).pvalue
        print(result["fragment_in_smiles"], result["temperature"], p_value)

        result["original_prediction_indices"] = [
            i for i, x in enumerate(res["predictions"]) if x in result["smiles"]
        ]
        result["expected"] = [
            res["expectations"][i] for i in result["original_prediction_indices"]
        ]
        result["p_value"] = p_value
        test2_smiles.append(result)

    compiled_res = {
        "fragment": fragment,
        "prevalence": prevalence_of_fragment,
        "test_results": test_2_res,
        "num_points": num_points,
        "temperatures": temperatures,
        "analysis": test2_smiles,
    }

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_pickle(f"{OUTDIR}/{timestr}_{fragment}.pkl", compiled_res)


if __name__ == "__main__":
    for fragment in FRAGMENTS:
        train_test(fragment, NUM_POINTS, TEMPERATURES)