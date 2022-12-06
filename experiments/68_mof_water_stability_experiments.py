from gpt3forchem.data import get_waterstability_data
from gpt3forchem.api_wrappers import fine_tune, query_gpt3, extract_prediction
from gpt3forchem.input import generate_water_stability_prompts, generate_water_stability_prompts_confidence
from gpt3forchem.helpers import make_if_not_exists
import time 
from fastcore.utils import save_pickle
from pycm import ConfusionMatrix

from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
TRAIN_SIZE = 0.8
OUTDIR = 'results/20221204-water-stability_downsampled_5'

make_if_not_exists(OUTDIR)

def train_test(random_state=None, use_confidence=False, downsample=1.0):
    df = get_waterstability_data()
    train, test = train_test_split(df, train_size=TRAIN_SIZE,  random_state=random_state, stratify=df['stability'])

    # downsample the "high" class on the training set
    train_high = train[train['stability'] == 'high']
    train_low = train[train['stability'] == 'low']
    train_high = train_high.sample(frac=downsample, random_state=random_state)
    train = pd.concat([train_high, train_low])

    if use_confidence:
        train_prompts = generate_water_stability_prompts_confidence(train)
        test_prompts = generate_water_stability_prompts_confidence(test)
    else:
        train_prompts = generate_water_stability_prompts(train)
        test_prompts = generate_water_stability_prompts(test)

    filename_base = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    train_filename = f"run_files/{filename_base}_train_prompts_water_stability_{TRAIN_SIZE}_{use_confidence}.jsonl"
    valid_filename = f"run_files/{filename_base}_valid_prompts_water_stability_{use_confidence}.jsonl"

    train_prompts.to_json(train_filename, orient="records", lines=True)
    test_prompts.to_json(valid_filename, orient="records", lines=True)

    modelname = fine_tune(train_filename, valid_filename)

    completions = query_gpt3(modelname, test_prompts)

    if not use_confidence:
        predictions =[int(extract_prediction(completions, i)) for i in range(len(completions['choices']))]
        true = test['stability'].apply(lambda x: 0 if x == 'low' else 1).values
        assert len(predictions) == len(true)
        confidence_cm = None
        high_confidence_cm = None
    else:
        extracted_completions_w_conf = [i['text'] for i in completions['choices']]
        predictions_w_conf = []
        confidence_w_conf = []

        for compl in extracted_completions_w_conf:
            compl = compl.split('@@@')[0]
            pred, conf = compl.split(' (')
            predictions_w_conf.append(int(pred))
            confidence_w_conf.append(conf[:-1])

        predictions = np.array(predictions_w_conf)
        confidence_w_conf = np.array(confidence_w_conf)
        confidence_cm = ConfusionMatrix(test['confidence'].values, [s.split()[0] for s in confidence_w_conf])
        confidence_mask = confidence_w_conf=='high confidence'
        true = test['stability'].apply(lambda x: 0 if x == 'low' else 1).values
        high_confidence_cm  = ConfusionMatrix(actual_vector=true[confidence_mask], predict_vector=predictions[confidence_mask])
        
    cm = ConfusionMatrix(true, predictions)
    outputs = {
        'train_size': TRAIN_SIZE,
        'with_confidence': use_confidence,
        'modelname': modelname,
        'train_filename': train_filename,
        'valid_filename': valid_filename,
        'completions': completions,
        'cm': cm, 
        'downsample_fraction': downsample,
        'confidence_cm': confidence_cm,
        'high_confidence_cm': high_confidence_cm
    }
    print(f"Using confidence {use_confidence} | Accuracy: {cm.ACC_Macro}, F1: {cm.F1_Macro}")

    save_pickle(f"{OUTDIR}/{filename_base}_{seed}_{use_confidence}_{downsample}_water_stability.pkl", outputs)

    return outputs

if __name__ == "__main__":
    for seed in range(10):
        seed += 42
        for use_confidence in [True, False]:
            for downsample in [0.8, 0.6, 0.3, 1.0]:
                outputs = train_test(seed, use_confidence)
