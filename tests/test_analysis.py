from gpt3forchem.analysis import string_distances


def test_string_distances():
    training_set = ["AAA", "BBB", "CCC"]
    query_string = "BBB"
    result = string_distances(training_set, query_string)

    assert result["NormalizedLevenshtein_min"] == 0.0
    assert result["NormalizedLevenshtein_max"] == 1.0
