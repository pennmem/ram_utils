import numpy as np
import pandas as pd
from traits.api import (
    Int, Float, String, Array,
    Dict, DictStrFloat, DictStrInt,
    ListBool, ListStr, ListInt, ListFloat
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
    item = ListStr(desc='list item (a.k.a. word)')
    session = ListInt(desc='session number')
    listno = ListInt(desc="item's list number")
    serialpos = ListInt(desc='item serial position')
    phase = ListStr(desc='list phase type (stim, non-stim, etc.)')

    # FIXME: these should allow Nones
    recognized = ListBool(desc='item in recognition subtask recognized')
    rejected = ListBool(desc='lure item in recognition subtask rejected')

    is_stim_list = ListBool(desc='item is from a stim list')
    is_post_stim_item = ListBool(desc='stimulation occurred on the previous item')
    is_stim_item = ListBool(desc='stimulation occurred on this item')
    recalled = ListBool(desc='item was recalled')
    thresh = ListFloat(desc='classifier threshold')

    prob = Array(dtype=np.float64, desc='probability of recall')


class StimSessionSummary(Summary):
    """Summary data specific to sessions with stimulation."""
    # FIXME: tags, regions can be nullable
    is_ps4_session = ListBool(desc='list is part of a PS4 session')
    stim_anode_tag = ListStr(desc='stim anode label')
    stim_cathode_tag = ListStr(desc='stim cathode label')
    region = ListStr(desc='brain region of stim pair')
    pulse_frequency = Array(dtype=np.float64, desc='stim pulse frequency [Hz]')
    amplitude = Array(dtype=np.float64, desc='stim amplitude [mA]')
    duration = Array(dtype=np.float64, desc='stim duration [ms]')
    burst_frequency = Array(dtype=np.float64, desc='stim burst frequency [Hz]')


# FIXME
# class CatFRSessionSummary(FRSessionSummary):
#     """Extends standard FR session summaries for categorized free recall
#     experiments.
#
#     """
#     irt_within_cat = Float(desc='average inter-response time within categories')
#     irt_between_cat = Float(desc='average inter-response time between categories')
