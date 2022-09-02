from sklearn.model_selection import train_test_split

from gpt3forchem.data import get_photoswitch_data
from gpt3forchem.input import generate_inverse_photoswitch_prompts

REPEATS = 10
TRAIN_TEST_RATIO = 0.8
