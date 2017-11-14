import numpy as np
import pandas as pd
from traits.api import (
    Int, Float, String, Array,
    Dict, DictStrFloat, DictStrInt,
    ListBool, ListStr,
)

from ramutils.schema import Schema


def DictStrArray(**kwargs):
    """Trait for a dict of numpy arrays.

    Keyword arguments
    -----------------
    dtype : np.dtype
        Array dtype (default: ``np.float64``)
    shape : list-like
        Shape for the array

    Notes
    -----
    All keyword arguments not specified above are passed on to the ``Dict``
    constructor.

    """
    kwargs['key_trait'] = String
    kwargs['value_trait'] = Array(dtype=kwargs.pop('dtype', np.float64),
                                  shape=kwargs.pop('shape', None))
    return Dict(**kwargs)


class Summary(Schema):
    """Base class for all summary objects."""
    # FIXME: only convert to DataFrame once?
    def to_dataframe(self):
        """Convert the summary to a :class:`pd.DataFrame` for easier
        manipulation.

        Returns
        -------
        pd.DataFrame

        """
        columns = {
            trait: getattr(self, trait)
            for trait in self.visible_traits()
        }
        return pd.DataFrame(columns)


class FRSessionSummary(Summary):
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
    prob_first_recall = Array(desc='probability of each encoding position being recalled first')

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
    """Extends standard FR session summaries for categorized free recall
    experiments.

    """
    irt_within_cat = Float(desc='average inter-response time within categories')
    irt_between_cat = Float(desc='average inter-response time between categories')


class StimSummary(Summary):
    """Stimulation-related summary of experiments.

    Notes
    -----
    All dicts use the stim pair label as the key.

    """
    frequency = DictStrFloat(desc='stimulation pulse frequency')
    amplitude = DictStrFloat(desc='stimulation amplitude')

    prob_recall = DictStrFloat(desc='average probability of recall by serial position')
    prob_stim_recall = DictStrFloat(desc='probability of recall by serial position for stim lists')
    prob_nostim_recall = DictStrFloat(desc='probability of recall by serial position for non-stim lists')
    prob_stim = DictStrFloat(desc='probability of stimulation by serial position')
    prob_first_stim_recall = DictStrArray(desc='probability of each encoding position being recalled first for stim lists')
    prob_first_nostim_recall = DictStrArray(desc='probability of each encoding position being recalled first for non-stim lists')

    # FIXME
    list_number = DictStrArray(desc='???')

    # FIXME: are these arrays?
    n_recalls_per_list = DictStrArray(dtype=np.int, desc="number of recalls by list")
    n_stims_per_list = DictStrArray(dtype=np.int, desc="number of recalls by stim list")

    # FIXME: are these more complicated than just lists? In FR6 they are oddly dicts
    is_baseline_list = ListBool(desc="masks which lists are baseline")
    is_nonstim_list = ListBool(desc="masks which lists are non-stim")
    is_stim_list = ListBool(desc="masks which lists are stim")
    is_ps_list = ListBool(desc="masks which lists are PS")

    n_correct_stim = DictStrInt(desc="total number of correctly recalled words for stim lists")
    n_total_stim = DictStrInt(desc="total number of stim events ???")  # FIXME: desc
    pc_from_stim = DictStrFloat(desc="percentage of correctly recalled stim words ???")  # FIXME: name, desc

    chisqr = DictStrFloat(desc='chi squared for stim vs. non-stim lists')
    pvalue = DictStrFloat(desc='p-value for stim vs. non-stim lists')
