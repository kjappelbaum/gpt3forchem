from gpt3forchem.polymers.create_prompts import create_single_property_forward_prompts
from gpt3forchem.polymers.data import get_data
import pandas as pd


def test_create_prompt():
    data = get_data()
    prompts = create_single_property_forward_prompts(data, "deltaGmin")
    assert isinstance(prompts, pd.DataFrame)
    assert "adsorption energy" in prompts["prompts"][0]
