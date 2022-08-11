import pandas as pd

ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE = "what is the {property} of {text}###"
ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE = " {value}@@@"

TARGET_RENAME_DICT = {
    "pce_bin": "PCE",
}


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
