from typing import Iterable, Callable, Dict, List
from collections import defaultdict
from strsimpy.levenshtein import Levenshtein
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from strsimpy.longest_common_subsequence import LongestCommonSubsequence
import numpy as np
import concurrent.futures


def string_distances(training_set: Iterable[str], query_string: str):

    distances = defaultdict(list)

    metrics = [
        ("Levenshtein", Levenshtein()),
        ("NormalizedLevenshtein", NormalizedLevenshtein()),
        ("LongestCommonSubsequence", LongestCommonSubsequence()),
    ]

    aggregations = [
        ("min", lambda x: np.min(x)),
        ("max", lambda x: np.max(x)),
        ("mean", lambda x: np.mean(x)),
        ("std", lambda x: np.std(x)),
    ]

    for training_string in training_set:
        for metric_name, metric in metrics:
            distances[metric_name].append(metric.distance(training_string, query_string))

    aggregtated_distances = {}

    for k, v in distances.items():
        for agg_name, agg_func in aggregations:
            aggregtated_distances[f"{k}_{agg_name}"] = agg_func(v)

    return aggregtated_distances
