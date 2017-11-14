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

    recalled = ListBool(desc='item was recalled')
    thresh = ListFloat(desc='classifier threshold')

    prob = Array(dtype=np.float64, desc='probability of recall')

    def populate(self, events):
        """Populate data from events.

        Parameters
        ----------
        events : np.recarray

        """
        self.item = events.item_name
        self.session = events.session
        self.listno = events.list
        self.serialpos = events.serialpos
        self.phase = events.phase
        self.recalled = events.recalled
        self.thresh = [0.5] * len(self.item)

        # FIXME: self.recognized
        # FIXME: self.rejected
        # FIXME: self.prob

# FIXME
# class CatFRSessionSummary(FRSessionSummary):
#     """Extends standard FR session summaries for categorized free recall
#     experiments.
#
#     """
#     irt_within_cat = Float(desc='average inter-response time within categories')
#     irt_between_cat = Float(desc='average inter-response time between categories')


class StimSessionSummary(Summary):
    """Summary data specific to sessions with stimulation."""
    is_stim_list = ListBool(desc='item is from a stim list')
    is_post_stim_item = ListBool(desc='stimulation occurred on the previous item')
    is_stim_item = ListBool(desc='stimulation occurred on this item')
    is_ps4_session = ListBool(desc='list is part of a PS4 session')

    # FIXME: tags, regions can be nullable
    stim_anode_tag = ListStr(desc='stim anode label')
    stim_cathode_tag = ListStr(desc='stim cathode label')
    region = ListStr(desc='brain region of stim pair')
    pulse_frequency = Array(dtype=np.float64, desc='stim pulse frequency [Hz]')
    amplitude = Array(dtype=np.float64, desc='stim amplitude [mA]')
    duration = Array(dtype=np.float64, desc='stim duration [ms]')

    def populate(self, events, is_ps4_session=False):
        """Populate stim data from events.

        Parameters
        ----------
        events : np.recarray
        is_ps4_session : bool
            Whether or not this experiment is also a PS4 session.

        """
        self.is_stim_list = [e.phase == 'STIM' for e in events]
        # self.is_post_stim_item = is_post_stim_item   # FIXME
        # self.is_stim_item = is_stim_item  # FIXME
        self.is_ps4_session = [is_ps4_session] * len(events)

        # FIXME: region

        # FIXME: amplitudes, etc. should be CSV for multistim
        self.stim_anode_tag = [e.stim_params.anode_label for e in events]
        self.stim_cathode_tag = [e.stim_params.cathode_label for e in events]
        self.pulse_frequency = [e.stim_params.pulse_freq for e in events]
        self.amplitude = [e.stim_params.amplitude for e in events]
        self.duration = [e.stim_params.stim_duration for e in events]


class FRStimSessionSummary(FRSessionSummary, StimSessionSummary):
    """Summary for FR sessions with stim."""
    def populate(self, events, is_ps4_session=False):
        FRSessionSummary.populate(self, events)
        StimSessionSummary.populate(self, events, is_ps4_session)
