# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/06_helpers.ipynb.

# %% auto 0
__all__ = ['HashableDataFrame', 'picp', 'multiclass_vote_to_probabilities', 'multiclass_brier_score',
           'expected_calibration_error', 'only_mode', 'augmented_classification_scores', 'make_if_not_exists']

# %% ../notebooks/06_helpers.ipynb 1
from functools import lru_cache
from hashlib import sha256
from typing import Iterable, Optional, Callable
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from collections import defaultdict
from collections import Counter
from pycm import ConfusionMatrix
from .input import encode_categorical_value
from scipy.stats import mode
import os

# %% ../notebooks/06_helpers.ipynb 2
# taken from https://gist.github.com/dsevero/3f3db7acb45d6cd8e945e8a32eaca168
class HashableDataFrame(pd.DataFrame):
    def __init__(self, obj):
        super().__init__(obj)

    def __hash__(self):
        hash_value = sha256(hash_pandas_object(self, index=True).values)
        hash_value = hash(hash_value.hexdigest())
        return hash_value

    def __eq__(self, other):
        return self.equals(other)

# %% ../notebooks/06_helpers.ipynb 9
# taken from https://github.com/IBM/UQ360/blob/main/uq360/metrics/regression_metrics.py
def picp(y_true, y_lower, y_upper):
    """
    Prediction Interval Coverage Probability (PICP). Computes the fraction of samples for which the grounds truth lies
    within predicted interval. Measures the prediction interval calibration for regression.
    Args:
        y_true: Ground truth
        y_lower: predicted lower bound
        y_upper: predicted upper bound
    Returns:
        float: the fraction of samples for which the grounds truth lies within predicted interval.
    """
    satisfies_upper_bound = y_true <= y_upper
    satisfies_lower_bound = y_true >= y_lower
    return np.mean(satisfies_upper_bound * satisfies_lower_bound)


# %% ../notebooks/06_helpers.ipynb 14
def multiclass_vote_to_probabilities(
        prediction_frame: pd.DataFrame, # input dataframe with predictions and representations 
        prediction_colum: str, # name of the column with predictions
        representation_column: str, # name of the column with representations
        classes: Iterable=np.arange(5) # names of all possible classes
    ) -> pd.DataFrame:
    """For each representation, get the frequency of each class."""
    
    new_frame = []
    reprs = prediction_frame[representation_column].unique()
    for repr in reprs:
        repr_frame = prediction_frame[prediction_frame[representation_column] == repr]
        value_counts = dict(repr_frame[prediction_colum].value_counts())
        frequencies = {}
        for class_ in classes:
            try:
                frequencies[class_] = value_counts.get(class_, 0) / len(repr_frame)
            except KeyError:
                frequencies[class_] = 0
            except ZeroDivisionError:
                frequencies[class_] = 0 

        frequencies[representation_column] = repr
        new_frame.append(frequencies)
    
    return pd.DataFrame(new_frame)

# %% ../notebooks/06_helpers.ipynb 21
# code taken from https://github.com/IBM/UQ360/blob/main/uq360/metrics/classification_metrics.py as having the tensorflow dependency is annoying
# The original LICENSE
#                               Apache License
#                         Version 2.0, January 2004
#                      http://www.apache.org/licenses/

# TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

# 1. Definitions.

#    "License" shall mean the terms and conditions for use, reproduction,
#    and distribution as defined by Sections 1 through 9 of this document.

#    "Licensor" shall mean the copyright owner or entity authorized by
#    the copyright owner that is granting the License.

#    "Legal Entity" shall mean the union of the acting entity and all
#    other entities that control, are controlled by, or are under common
#    control with that entity. For the purposes of this definition,
#    "control" means (i) the power, direct or indirect, to cause the
#    direction or management of such entity, whether by contract or
#    otherwise, or (ii) ownership of fifty percent (50%) or more of the
#    outstanding shares, or (iii) beneficial ownership of such entity.

#    "You" (or "Your") shall mean an individual or Legal Entity
#    exercising permissions granted by this License.

#    "Source" form shall mean the preferred form for making modifications,
#    including but not limited to software source code, documentation
#    source, and configuration files.

#    "Object" form shall mean any form resulting from mechanical
#    transformation or translation of a Source form, including but
#    not limited to compiled object code, generated documentation,
#    and conversions to other media types.

#    "Work" shall mean the work of authorship, whether in Source or
#    Object form, made available under the License, as indicated by a
#    copyright notice that is included in or attached to the work
#    (an example is provided in the Appendix below).

#    "Derivative Works" shall mean any work, whether in Source or Object
#    form, that is based on (or derived from) the Work and for which the
#    editorial revisions, annotations, elaborations, or other modifications
#    represent, as a whole, an original work of authorship. For the purposes
#    of this License, Derivative Works shall not include works that remain
#    separable from, or merely link (or bind by name) to the interfaces of,
#    the Work and Derivative Works thereof.

#    "Contribution" shall mean any work of authorship, including
#    the original version of the Work and any modifications or additions
#    to that Work or Derivative Works thereof, that is intentionally
#    submitted to Licensor for inclusion in the Work by the copyright owner
#    or by an individual or Legal Entity authorized to submit on behalf of
#    the copyright owner. For the purposes of this definition, "submitted"
#    means any form of electronic, verbal, or written communication sent
#    to the Licensor or its representatives, including but not limited to
#    communication on electronic mailing lists, source code control systems,
#    and issue tracking systems that are managed by, or on behalf of, the
#    Licensor for the purpose of discussing and improving the Work, but
#    excluding communication that is conspicuously marked or otherwise
#    designated in writing by the copyright owner as "Not a Contribution."

#    "Contributor" shall mean Licensor and any individual or Legal Entity
#    on behalf of whom a Contribution has been received by Licensor and
#    subsequently incorporated within the Work.

# 2. Grant of Copyright License. Subject to the terms and conditions of
#    this License, each Contributor hereby grants to You a perpetual,
#    worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#    copyright license to reproduce, prepare Derivative Works of,
#    publicly display, publicly perform, sublicense, and distribute the
#    Work and such Derivative Works in Source or Object form.

# 3. Grant of Patent License. Subject to the terms and conditions of
#    this License, each Contributor hereby grants to You a perpetual,
#    worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#    (except as stated in this section) patent license to make, have made,
#    use, offer to sell, sell, import, and otherwise transfer the Work,
#    where such license applies only to those patent claims licensable
#    by such Contributor that are necessarily infringed by their
#    Contribution(s) alone or by combination of their Contribution(s)
#    with the Work to which such Contribution(s) was submitted. If You
#    institute patent litigation against any entity (including a
#    cross-claim or counterclaim in a lawsuit) alleging that the Work
#    or a Contribution incorporated within the Work constitutes direct
#    or contributory patent infringement, then any patent licenses
#    granted to You under this License for that Work shall terminate
#    as of the date such litigation is filed.

# 4. Redistribution. You may reproduce and distribute copies of the
#    Work or Derivative Works thereof in any medium, with or without
#    modifications, and in Source or Object form, provided that You
#    meet the following conditions:

#    (a) You must give any other recipients of the Work or
#        Derivative Works a copy of this License; and

#    (b) You must cause any modified files to carry prominent notices
#        stating that You changed the files; and

#    (c) You must retain, in the Source form of any Derivative Works
#        that You distribute, all copyright, patent, trademark, and
#        attribution notices from the Source form of the Work,
#        excluding those notices that do not pertain to any part of
#        the Derivative Works; and

#    (d) If the Work includes a "NOTICE" text file as part of its
#        distribution, then any Derivative Works that You distribute must
#        include a readable copy of the attribution notices contained
#        within such NOTICE file, excluding those notices that do not
#        pertain to any part of the Derivative Works, in at least one
#        of the following places: within a NOTICE text file distributed
#        as part of the Derivative Works; within the Source form or
#        documentation, if provided along with the Derivative Works; or,
#        within a display generated by the Derivative Works, if and
#        wherever such third-party notices normally appear. The contents
#        of the NOTICE file are for informational purposes only and
#        do not modify the License. You may add Your own attribution
#        notices within Derivative Works that You distribute, alongside
#        or as an addendum to the NOTICE text from the Work, provided
#        that such additional attribution notices cannot be construed
#        as modifying the License.

#    You may add Your own copyright statement to Your modifications and
#    may provide additional or different license terms and conditions
#    for use, reproduction, or distribution of Your modifications, or
#    for any such Derivative Works as a whole, provided Your use,
#    reproduction, and distribution of the Work otherwise complies with
#    the conditions stated in this License.

# 5. Submission of Contributions. Unless You explicitly state otherwise,
#    any Contribution intentionally submitted for inclusion in the Work
#    by You to the Licensor shall be under the terms and conditions of
#    this License, without any additional terms or conditions.
#    Notwithstanding the above, nothing herein shall supersede or modify
#    the terms of any separate license agreement you may have executed
#    with Licensor regarding such Contributions.

# 6. Trademarks. This License does not grant permission to use the trade
#    names, trademarks, service marks, or product names of the Licensor,
#    except as required for reasonable and customary use in describing the
#    origin of the Work and reproducing the content of the NOTICE file.

# 7. Disclaimer of Warranty. Unless required by applicable law or
#    agreed to in writing, Licensor provides the Work (and each
#    Contributor provides its Contributions) on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#    implied, including, without limitation, any warranties or conditions
#    of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#    PARTICULAR PURPOSE. You are solely responsible for determining the
#    appropriateness of using or redistributing the Work and assume any
#    risks associated with Your exercise of permissions under this License.

# 8. Limitation of Liability. In no event and under no legal theory,
#    whether in tort (including negligence), contract, or otherwise,
#    unless required by applicable law (such as deliberate and grossly
#    negligent acts) or agreed to in writing, shall any Contributor be
#    liable to You for damages, including any direct, indirect, special,
#    incidental, or consequential damages of any character arising as a
#    result of this License or out of the use or inability to use the
#    Work (including but not limited to damages for loss of goodwill,
#    work stoppage, computer failure or malfunction, or any and all
#    other commercial damages or losses), even if such Contributor
#    has been advised of the possibility of such damages.

# 9. Accepting Warranty or Additional Liability. While redistributing
#    the Work or Derivative Works thereof, You may choose to offer,
#    and charge a fee for, acceptance of support, warranty, indemnity,
#    or other liability obligations and/or rights consistent with this
#    License. However, in accepting such obligations, You may act only
#    on Your own behalf and on Your sole responsibility, not on behalf
#    of any other Contributor, and only if You agree to indemnify,
#    defend, and hold each Contributor harmless for any liability
#    incurred by, or claims asserted against, such Contributor by reason
#    of your accepting any such warranty or additional liability.

# END OF TERMS AND CONDITIONS

# APPENDIX: How to apply the Apache License to your work.

#    To apply the Apache License to your work, attach the following
#    boilerplate notice, with the fields enclosed by brackets "[]"
#    replaced with your own identifying information. (Don't include
#    the brackets!)  The text should be enclosed in the appropriate
#    comment syntax for the file format. We also recommend that a
#    file or class name and description of purpose be included on the
#    same "printed page" as the copyright notice for easier
#    identification within third-party archives.

# Copyright [yyyy] [name of copyright owner]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def multiclass_brier_score(y_true, y_prob):
    """Brier score for multi-class.

    Args:
        y_true: array-like of shape (n_samples,)
            ground truth labels.
        y_prob: array-like of shape (n_samples, n_classes).
            Probability scores from the base model.
    Returns:
        float: Brier score.
    """
    assert len(y_prob.shape) > 1, "y_prob should be array-like of shape (n_samples, n_classes)"

    y_target = np.zeros_like(y_prob)
    y_target[np.arange(y_true.size), y_true] = 1.0
    return np.mean(np.sum((y_target - y_prob) ** 2, axis=1))

def expected_calibration_error(y_true, y_prob, y_pred=None, num_bins=10, return_counts=False):
    """ Computes the reliability curve and the  expected calibration error [1]_ .

    References:
        .. [1] Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger; Proceedings of the 34th International Conference
            on Machine Learning, PMLR 70:1321-1330, 2017.

    The expected calibration error is the difference in expectation between the confidence and accuracy. 

    Args:
        y_true: array-like of shape (n_samples,)
            ground truth labels.
        y_prob: array-like of shape (n_samples, n_classes).
            Probability scores from the base model.
        y_pred: array-like of shape (n_samples,)
            predicted labels.
        num_bins: number of bins.
        return_counts: set to True to return counts also.

    Returns:
        float or tuple:
            - ece (float): expected calibration error.
            - confidences_in_bins: average confidence in each bin (returned only if return_counts is True).
            - accuracies_in_bins: accuracy in each bin (returned only if return_counts is True).
            - frac_samples_in_bins: fraction of samples in each bin (returned only if return_counts is True).
    """

    assert len(y_prob.shape) > 1, "y_prob should be array-like of shape (n_samples, n_classes)"
    num_samples, num_classes = y_prob.shape
    top_scores = np.max(y_prob, axis=1)

    if y_pred is None:
        y_pred = np.argmax(y_prob, axis=1)

    if num_classes == 2:
        bins_edges = np.histogram_bin_edges([], bins=num_bins, range=(0.5, 1.0))
    else:
        bins_edges = np.histogram_bin_edges([], bins=num_bins, range=(0.0, 1.0))

    non_boundary_bin_edges = bins_edges[1:-1]
    bin_centers = (bins_edges[1:] + bins_edges[:-1])/2

    sample_bin_ids = np.digitize(top_scores, non_boundary_bin_edges)

    num_samples_in_bins = np.zeros(num_bins)
    accuracies_in_bins = np.zeros(num_bins)
    confidences_in_bins = np.zeros(num_bins)

    for bin in range(num_bins):
        num_samples_in_bins[bin] = len(y_pred[sample_bin_ids == bin])
        if num_samples_in_bins[bin] > 0:
            accuracies_in_bins[bin] = np.sum(y_true[sample_bin_ids == bin] == y_pred[sample_bin_ids == bin]) / num_samples_in_bins[bin]
            confidences_in_bins[bin] = np.sum(top_scores[sample_bin_ids == bin]) / num_samples_in_bins[bin]

    ece = np.sum(
        num_samples_in_bins * np.abs(accuracies_in_bins - confidences_in_bins) / num_samples
    )
    frac_samples_in_bins = num_samples_in_bins / num_samples

    if not return_counts:
        return ece
    else:
        return ece, confidences_in_bins, accuracies_in_bins, frac_samples_in_bins, bin_centers



# %% ../notebooks/06_helpers.ipynb 27
def only_mode(x):
    return mode(x)[0][0]

# %% ../notebooks/06_helpers.ipynb 28
def augmented_classification_scores(repr, true, predictions, cat_encode_func: Optional[Callable]=encode_categorical_value, class_names=np.arange(5)): 
    augmented_predictions = pd.DataFrame(
        {
            'repr': repr,
            'true': true,
            'predictions': predictions,
        })
    
    if cat_encode_func is not None:
        augmented_predictions['true'] = augmented_predictions['true'].apply(cat_encode_func)
        augmented_predictions['predictions'] = augmented_predictions['predictions'].apply(cat_encode_func)
    
    predictions_augmented = augmented_predictions.groupby('repr').agg(['mean', 'std', only_mode])
    cm = ConfusionMatrix(
        predictions_augmented['true']['only_mode'].values,
        predictions_augmented['predictions']['only_mode'].values,
    )
    
    class_probablities = multiclass_vote_to_probabilities(augmented_predictions, 'predictions', 'repr')

    brier_score = multiclass_brier_score(augmented_predictions.groupby('repr').mean()['true'].values.astype(int), class_probablities[class_names].values)

    ece = expected_calibration_error(augmented_predictions.groupby('repr').mean()['true'].values.astype(int), class_probablities[class_names].values)

    return cm, brier_score, ece


# %% ../notebooks/06_helpers.ipynb 30
def make_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
