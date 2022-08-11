import pandas as pd
from collections import Counter

ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE = "what is the {property} of {text}###"
ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE = " {value}@@@"

TARGET_RENAME_DICT = {
    "logKH_CO2_cat": "logarithmic Henry coefficient for CO2",
    "logKH_CO2": "logarithmic Henry coefficient for CO2",
}


def create_single_property_forward_prompts(df, target, text):
    prompts = []

    target_name = target
    for key, value in TARGET_RENAME_DICT.items():
        target_name = target_name.replace(key, value)

    for _, row in df.iterrows():
        # in contrast to the polymers, we now be default use ints for the encoding
        value = row[target]

        if not "ERROR" in row[text]:
            name = row[text].split(";")[0].replace(" MOFid-v1", "").encode("utf-8")
            prompts.append(
                {
                    "prompt": ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE.format(
                        property=target_name, text=name
                    ),
                    "completion": ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE.format(value=value),
                }
            )

    return pd.DataFrame(prompts)
