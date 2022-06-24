import pandas as pd

ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE = "what is the {property} of {text}###"
ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE = " {value}@@@"

TARGET_RENAME_DICT = {
    "A2_normalized_cat": "virial coefficient",
    "deltaGmin_cat": "adsorption energy",
    "A2_normalized": "virial coefficient",
    "deltaGmin": "adsorption energy",
}


# this code is stupid. Use a bidict
def encode_categorical_value(value):
    if value == "very small":
        return 0
    elif value == "small":
        return 1
    elif value == "medium":
        return 2
    elif value == "large":
        return 3
    elif value == "very large":
        return 4
    else:
        raise ValueError("Unknown value: %s" % value)


def decode_categorical_value(value):
    if value == "0":
        return "very small"
    elif value == "1":
        return "small"
    elif value == "2":
        return "medium"
    elif value == "3":
        return "large"
    elif value == "4":
        return "very large"
    else:
        raise ValueError("Unknown value: %s" % value)


def create_single_property_forward_prompts(df, target, encode_value=True):
    prompts = []

    target_name = target
    for key, value in TARGET_RENAME_DICT.items():
        target_name = target_name.replace(key, value)

    for _, row in df.iterrows():
        if encode_value:
            value = encode_categorical_value(row[target])
        else:
            value = row[target]

        prompts.append(
            {
                "prompt": ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE.format(
                    property=target_name, text=row["string"]
                ),
                "completion": ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE.format(value=value),
            }
        )

    return pd.DataFrame(prompts)
