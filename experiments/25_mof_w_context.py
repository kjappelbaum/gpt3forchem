import time
from collections import Counter

import numpy as np
from pycm import ConfusionMatrix
from sklearn.model_selection import train_test_split
import pandas as pd
from gpt3forchem.api_wrappers import extract_prediction, fine_tune, query_gpt3
from gpt3forchem.data import discretize, get_mof_data
from gpt3forchem.input import (
    create_single_property_forward_prompts,
    create_single_property_forward_prompts_multiple_targets,
)
from sklearn.dummy import DummyClassifier

# We can try different experiments for adding context
# to the prompts. We can try:
# 1. Adding the gas name to the prompt
# 2. Adding the gas name and formula to the prompt
# 3. Adding the gas name, formula and some physical descriptors to the prompt
#
# We can additionaly play with
# Adding a different number of other gases, and in particular different types (e.g. polar, non-polar)
#
# We will also need at least two points of reference
# 1. Performance of dummy models (Lower bound)
# 2. Performance of models trained directly on water (Upper bound)
# 3. XGboost Classifiers with phyiscal descriptors for the gasses
