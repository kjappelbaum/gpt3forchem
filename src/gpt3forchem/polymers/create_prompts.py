import pandas as pd
from collections import Counter

ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE = "what is the {property} of {text}###"
ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE = " {value}@@@"

ONE_PROPERTY_INVERSE_PROMPT_TEMPLATE_CAT = "what is a polymer with {class_name} {property}?###"
ONE_PROPERTY_INVERSE_COMPLETION_TEMPLATE_CAT = " {text}@@@"

ONE_PROPERTY_INVERSE_PROMPT_TEMPLATE_CAT_W_COMPOSITION = "what is a polymer with {class_name} {property} and {num_A} A, {num_B} B, {num_W} W, and {num_R} R?###"

TARGET_RENAME_DICT = {
    "A2_normalized_cat": "virial coefficient",
    "deltaGmin_cat": "adsorption energy",
    "A2_normalized": "virial coefficient",
    "deltaGmin": "adsorption energy",
}


def _get_composition_dict(row):
    composition = Counter(row["string"].split("-"))
    comp_dict = {}
    for key in ["A", "B", "R", "W"]:
        try:
            count = composition[key]
        except KeyError:
            count = 0
        comp_dict[f"num_{key}"] = count
    return comp_dict


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
                "completion": ONE_PROPERTY_INVERSE_COMPLETION_TEMPLATE_CAT.format(value=value),
            }
        )

    return pd.DataFrame(prompts)


def create_single_property_inverse_prompts(df, target, encode_value=True, with_composition=True):
    prompts = []

    target_name = target
    for key, value in TARGET_RENAME_DICT.items():
        target_name = target_name.replace(key, value)

    for _, row in df.iterrows():
        if encode_value:
            value = encode_categorical_value(row[target])
        else:
            value = row[target]

        if with_composition:
            comp_dict = _get_composition_dict(row)

            prompt = ONE_PROPERTY_INVERSE_PROMPT_TEMPLATE_CAT_W_COMPOSITION.format(
                class_name=value, property=target_name, **comp_dict
            )
        else:
            prompt = (
                ONE_PROPERTY_INVERSE_PROMPT_TEMPLATE_CAT.format(
                    class_name=value, property=target_name
                ),
            )
        prompts.append(
            {
                "prompt": prompt,
                "completion": ONE_PROPERTY_INVERSE_COMPLETION_TEMPLATE_CAT.format(
                    text=row["string"]
                ),
            }
        )

    return pd.DataFrame(prompts)
