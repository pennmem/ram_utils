"""Schema definitions for files to be ingested by web reporting."""

import numpy as np
import h5py
from traits.api import HasTraits, Int, Float, String, Array


class _Schema(HasTraits):
    """Base class for defining serializable schema."""
    pass


class SessionSummary(_Schema):
    number = Int(desc='session number')
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
    prob_recall = Float(desc='probability of recall')
    prob_first_recall = Float(desc='probability of first recall?')
    n_math = Int(desc='number of math problems')
    n_correct_math = Int(desc='number of correctly answered math problems')
    pc_correct_math = Float(desc='percentage of correctly answered math problems')
    math_per_list = Float(desc='mean number of math problems per list')

    # FIXME: what are these?
    # irt_within_cat = None
    # irt_between_cat = None

    auc = Float(desc='classifier AUC')

    # FIXME: what are these?
    # ltt = None
    # fpr = None
    # tpr = None

    pc_diff_from_mean = Float(desc='percentage difference from mean')
    perm_AUCs = Float(desc='permutation test AUCs')
    perm_test_pvalue = None
    jstat_thresh = None
    jstat_percentile = None
    repetition_ratio = None


if __name__ == "__main__":
    summary = SessionSummary()
