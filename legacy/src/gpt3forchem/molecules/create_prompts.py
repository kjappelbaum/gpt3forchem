import pandas as pd

ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE = "what is the {property} of {text}###"
ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE = " {value}@@@"

TARGET_RENAME_DICT = {
    "logP_binned": "logP",
}

INVERSE_PROMPT_START = "What is a molecule with "
INVERSE_PROMPT_END = " and a logP of {}?###"


def create_single_property_forward_prompts(df, target, text):
    prompts = []

    target_name = target
    for key, value in TARGET_RENAME_DICT.items():
        target_name = target_name.replace(key, value)

    for _, row in df.iterrows():
        # in contrast to the polymers, we now be default use ints for the encoding
        value = row[target]
        name = row[text]
        prompts.append(
            {
                "prompt": ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE.format(
                    property=target_name, text=name
                ),
                "completion": ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE.format(value=value),
            }
        )

    return pd.DataFrame(prompts)


def create_inverse_prompt_row(row, composition=["C", "H", "O", "N", "P", "S"]):
    composition_string = ", ".join([f"{int(row[element])} {element}" for element in composition])
    return INVERSE_PROMPT_START + composition_string + INVERSE_PROMPT_END.format(row["logP_binned"])


def create_inverse_prompts(
    df,
    composition=["C", "H", "O", "N", "P", "S"],
    completion_len_cutoff=300,
    min_logP=-5,
    max_logP=15,
):
    prompts = []
    for _, row in df.iterrows():
        prompt = create_inverse_prompt_row(row, composition)
        completion = " {}@@@".format(row["smiles"])
        if len(completion) < completion_len_cutoff:
            if row["logP"] >= min_logP and row["logP"] <= max_logP:
                prompts.append({"prompt": prompt, "completion": completion})
    return pd.DataFrame(prompts)
