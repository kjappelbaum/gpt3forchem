import pandas as pd

ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE = "what is the {property} of {text}###"
ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE = " {value}@@@"

TARGET_RENAME_DICT = {"A2_normalized": "virial coefficient", "deltaGmin": "adsorption energy"}


def create_single_property_forward_prompts(df, target):
    prompts = []

    target_name = target
    for key, value in TARGET_RENAME_DICT.items():
        target_name = target_name.replace(key, value)

    for _, row in df.iterrows():
        prompts.append(
            {
                "prompt": ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE.format(
                    property=target_name, text=row["text"]
                ),
                "completion": ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE.format(value=row[target]),
            }
        )

    return pd.DataFrame(prompts)
