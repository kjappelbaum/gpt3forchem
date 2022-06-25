from click import prompt
from gpt3forchem.polymers.create_prompts import (
    create_single_property_forward_prompts,
    create_single_property_inverse_prompts,
)
from gpt3forchem.polymers.data import get_data
import pandas as pd


def test_create_prompt():
    data = get_data()
    prompts = create_single_property_forward_prompts(data, "deltaGmin")
    assert isinstance(prompts, pd.DataFrame)
    assert "adsorption energy" in prompts["prompt"][0]


def test_create_prompt():
    data = get_data()
    prompts = create_single_property_inverse_prompts(data, "deltaGmin_cat")
    assert isinstance(prompts, pd.DataFrame)
    assert "adsorption energy" in prompts["prompt"][0]
    assert "-" in prompts["completion"][0]
