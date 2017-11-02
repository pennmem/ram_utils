import numpy as np
from traits.api import Int, Float, String, Array

from ramutils.schema import Schema


class FRSessionSummary(Schema):
    """Free recall session summary data."""
    number = Int(desc='session number')  # FIXME: not seemingly used
    name = String(desc='experiment name')
    start = Float(desc='start timestamp')
    end = Float(desc='end timestamp')

    n_words = Int(desc='number of words')
    n_correct_words = Int(desc='number of correctly recalled words')
    pc_correct_words = Float(desc='percentage of correctly recalled words')

    n_pli = Int(desc='number of prior-list intrusions')
    pc_pli = Float(desc='percentage of prior-list intrusions')
    n_eli = Int(desc='number of extra-list intrusions')
    pc_eli = Float(desc='percentage of extra-list intrusions')

    prob_recall = Array(dtype=np.float64, desc='probability of recall by encoding position')
    prob_first_recall = Array(desc='probability of encoding position being recalled first')

    n_math = Int(desc='number of math problems')
    n_correct_math = Int(desc='number of correctly answered math problems')
    pc_correct_math = Float(desc='percentage of correctly answered math problems')
    math_per_list = Float(desc='mean number of math problems per list')

    auc = Float(desc='classifier AUC')
    fpr = Array(dtype=np.float64, desc='false positive rate')
    tpr = Array(dtype=np.float64, desc='true positive rate')

    pc_diff_from_mean = Array(dtype=np.float64, desc='percentage difference from mean')
    perm_AUCs = Array(dtype=np.float64, desc='permutation test AUCs')
    perm_test_pvalue = Float(desc='permutation test p-value')
    jstat_thresh = Float(desc='J statistic')
    jstat_percentile = Float(desc='J statistic percentile')


class CatFRSessionSummary(FRSessionSummary):
    """Extends standard FR session summaries for categoried free recall
    experiments.

    """
    irt_within_cat = Float(desc='average inter-response time within categories')
    irt_between_cat = Float(desc='average inter-response time between categories')
