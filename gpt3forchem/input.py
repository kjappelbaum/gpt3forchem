# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/03_input.ipynb.

# %% auto 0
__all__ = ['ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE', 'ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE',
           'POLYMER_ONE_PROPERTY_INVERSE_PROMPT_TEMPLATE_CAT', 'POLYMER_ONE_PROPERTY_INVERSE_COMPLETION_TEMPLATE_CAT',
           'POLYMER_ONE_PROPERTY_INVERSE_PROMPT_TEMPLATE_CAT_W_COMPOSITION', 'encode_categorical_value',
           'decode_categorical_value', 'get_polymer_composition_dict']

# %% ../notebooks/03_input.ipynb 2
_DEFAULT_ENCODING_DICT = {
    "very small": 0,
    "small": 1,
    "medium": 2,
    "large": 3,
    "very large": 4,
}

_DEFAULT_DECODING_DICT = {v: k for k, v in _DEFAULT_ENCODING_DICT.items()}


def encode_categorical_value(value, encoding_dict):
    try:
        return encoding_dict[value]
    except KeyError:
        raise ValueError("Unknown value: %s" % value)


def decode_categorical_value(value, decoding_dict):
    try:
        return decoding_dict[value]
    except KeyError:
        raise ValueError("Unknown value: %s" % value)


# %% ../notebooks/03_input.ipynb 3
ONE_PROPERTY_FORWARD_PROMPT_TEMPLATE = "what is the {property} of {text}###"
ONE_PROPERTY_FORWARD_COMPLETION_TEMPLATE = " {value}@@@"


# %% ../notebooks/03_input.ipynb 8
POLYMER_ONE_PROPERTY_INVERSE_PROMPT_TEMPLATE_CAT = (
    "what is a polymer with {class_name} {property}?###"
)
POLYMER_ONE_PROPERTY_INVERSE_COMPLETION_TEMPLATE_CAT = " {text}@@@"

POLYMER_ONE_PROPERTY_INVERSE_PROMPT_TEMPLATE_CAT_W_COMPOSITION = "what is a polymer with {class_name} {property} and {num_A} A, {num_B} B, {num_W} W, and {num_R} R?###"


# %% ../notebooks/03_input.ipynb 9
def get_polymer_composition_dict(row):
    composition = Counter(row["string"].split("-"))
    comp_dict = {}
    for key in ["A", "B", "R", "W"]:
        try:
            count = composition[key]
        except KeyError:
            count = 0
        comp_dict[f"num_{key}"] = count
    return comp_dict

