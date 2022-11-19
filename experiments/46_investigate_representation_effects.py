from gpt3forchem.helpers import selfies_to_one_hot, selfies_to_padded_selfies
from gpt3forchem.data import get_photoswitch_data
from sklearn.model_selection import train_test_split
from gpt3forchem.api_wrappers import extract_prediction, fine_tune, query_gpt3

from gpt3forchem.helpers import get_fragment_one_hot_mapping, selfies_to_padded_selfies, selfies_to_one_hot
from gpt3forchem.input import generate_one_hot_encoded_fragment_prompt, generate_fragment_prompt, create_single_property_forward_prompts

import time
from pycm import ConfusionMatrix
from fastcore.helpers import save_pickle
import click

OUTDIR = 'results/20221117-photoswitch_representation_effects'

def learning_curve_point(representation, train_set_size):
    data = get_photoswitch_data()
    if representation == 'selfies-one-hot':
        data['seflies_one_hot'] = selfies_to_one_hot(data['selfies'])
    elif 'pad' in representation:
        pad_length = int(representation.split('-')[-1])
        data['selfies_padded'] = selfies_to_padded_selfies(data['selfies'], pad_length)

    df_train, df_test = train_test_split(
        data, train_size=train_set_size, random_state=None, stratify=data["wavelength_cat"]
    )
    if representation == 'fragments':
        train_frame = generate_fragment_prompt(df_train, target='wavelength_cat', regression=False)
        test_frame = generate_fragment_prompt(df_test, target='wavelength_cat', regression=False)

    elif representation == 'fragments-one-hot':
        one_hot_mapper = get_fragment_one_hot_mapping(data['SMILES'])
        train_frame = generate_one_hot_encoded_fragment_prompt(df_train, target='wavelength_cat', regression=False, one_hot_mapper=one_hot_mapper)

        test_frame = generate_one_hot_encoded_fragment_prompt(df_test, target='wavelength_cat', regression=False, one_hot_mapper=one_hot_mapper)

    elif representation == 'selfies-one-hot':
        train_frame = create_single_property_forward_prompts(
            df_train,
            "wavelength_cat",
            {"wavelength_cat": "transition wavelength"},
            representation_col="seflies_one_hot",
        )

        test_frame = create_single_property_forward_prompts(
            df_test,
            "wavelength_cat",
            {"wavelength_cat": "transition wavelength"},
            representation_col="seflies_one_hot",
        )

    elif 'pad' in representation:
        train_frame = create_single_property_forward_prompts(
            df_train,
            "wavelength_cat",
            {"wavelength_cat": "transition wavelength"},
            representation_col="selfies_padded",
        )

        test_frame = create_single_property_forward_prompts(
            df_test,
            "wavelength_cat",
            {"wavelength_cat": "transition wavelength"},
            representation_col="selfies_padded",
        )

    else: 
        train_frame = create_single_property_forward_prompts(
            df_train,
            "wavelength_cat",
            {"wavelength_cat": "transition wavelength"},
            representation_col="SMILES",
        )

        test_frame = create_single_property_forward_prompts(
            df_test,
            "wavelength_cat",
            {"wavelength_cat": "transition wavelength"},
            representation_col="SMILES",
        )

    train_size = len(train_frame)
    test_size = len(test_frame)

    filename_base = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    train_filename = f"run_files/{filename_base}_train_prompts_photoswitch_{train_size}_{representation}.jsonl"
    valid_filename = f"run_files/{filename_base}_valid_prompts_photoswitch_{test_size}_{representation}.jsonl"

    train_frame.to_json(train_filename, orient="records", lines=True)
    test_frame.to_json(valid_filename, orient="records", lines=True)

    print(f"Training model on {train_size} training examples")
    modelname = fine_tune(train_filename, valid_filename)

    completions = query_gpt3(modelname, test_frame)
    predictions = [
        extract_prediction(completions, i)
        for i, completion in enumerate(completions["choices"])
    ]
    true = [
        test_frame.iloc[i]["completion"].split("@")[0]
        for i in range(len(predictions))
    ]
    assert len(predictions) == len(true)
    cm = ConfusionMatrix(true, predictions)

    results = {
        "train_set_size": train_set_size,
        "train_size": train_size,
        "test_size": test_size,
        "cm": cm,
        "accuracy": cm.ACC_Macro,
        "completions": completions,
        "train_filename": train_filename,
        "valid_filename": valid_filename,
        "modelname": modelname,
        'representation': representation,
        'padding_length': pad_length if 'pad' in representation else None,
    }

    outname = f"{OUTDIR}/{filename_base}_results_photoswitch_{train_size}_{representation}.pkl"

    save_pickle(outname, results)
    return results

REPRESENTATIONS = [
    # 'selfies',
    # 'fragments',
    # 'fragments-one-hot',
    # 'selfies-one-hot',
    'pad-3',
    'pad-4',
    'pad-5',
    'pad-6',
]

NUM_REPS = 10

TRAINING_SIZE = 50

@click.command()
@click.option('--representation', default=None)
@click.option('--train_set_size', type=int, default=None)
@click.option('--run_all', is_flag=True, default=False)
def main(representation, train_set_size, run_all):
    if not run_all:
        learning_curve_point(representation, train_set_size)
    else: 
        for _ in range(NUM_REPS):
            for representation in REPRESENTATIONS:
                    learning_curve_point(representation, TRAINING_SIZE)

if __name__ == '__main__':
    main()