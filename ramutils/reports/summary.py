import numpy as np
import pandas as pd
from traits.api import (
    String, Array, Dict, ListBool
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
    def to_dataframe(self, recreate=False):
        """Convert the summary to a :class:`pd.DataFrame` for easier
        manipulation.

        Keyword arguments
        -----------------
        recreate : bool
            Force re-creating the dataframe. Otherwise, it will only be created
            the first time this method is called and stored as an instance
            attribute.

        Returns
        -------
        pd.DataFrame

        """
        if not hasattr(self, '_df') or recreate:
            columns = {
                trait: getattr(self, trait)
                for trait in self.visible_traits()
                if trait not in ['rejected', 'region']  # FIXME
            }
            self._df = pd.DataFrame(columns)
        return self._df


class FRSessionSummary(Summary):
    """Free recall session summary data."""
    # FIXME: string dtypes for arrays
    item = Array(desc='list item (a.k.a. word)')
    session = Array(dtype=int, desc='session number')
    listno = Array(dtype=int, desc="item's list number")
    serialpos = Array(dtype=int, desc='item serial position')
    phase = Array(desc='list phase type (stim, non-stim, etc.)')

    recalled = Array(dtype=bool, desc='item was recalled')
    thresh = Array(dtype=np.float64, desc='classifier threshold')

    prob = Array(dtype=np.float64, desc='predicted probability of recall')

    def populate(self, events, recall_probs=None):
        """Populate data from events.

        Parameters
        ----------
        events : np.recarray
        recall_probs : np.ndarray
            Predicted probabilities of recall per item. If not given, assumed
            there is no relevant classifier and values of -999 are used to
            indicate this.

        """
        self.item = events.item_name
        self.session = events.session
        self.listno = events.list
        self.serialpos = events.serialpos
        self.phase = events.phase
        self.recalled = events.recalled
        self.thresh = [0.5] * len(events)
        self.prob = recall_probs if recall_probs is not None else [-999] * len(events)

    @property
    def num_lists(self):
        """Returns the total number of lists."""
        return len(self.to_dataframe().listno.unique())


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
    stim_anode_tag = Array(desc='stim anode label')
    stim_cathode_tag = Array(desc='stim cathode label')
    region = Array(desc='brain region of stim pair')
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
    def populate(self, events, recall_probs=None, is_ps4_session=False):
        FRSessionSummary.populate(self, events, recall_probs)
        StimSessionSummary.populate(self, events, is_ps4_session)

    @property
    def num_nonstim_lists(self):
        """Returns the number of non-stim lists."""
        df = self.to_dataframe()
        count = 0
        for listno in df.listno.unique():
            if not df[df.listno == listno].is_stim_list.all():
                count += 1
        return count

    @property
    def num_stim_lists(self):
        """Returns the number of stim lists."""
        df = self.to_dataframe()
        count = 0
        for listno in df.listno.unique():
            if df[df.listno == listno].is_stim_list.all():
                count += 1
        return count


class FR5SessionSummary(FRStimSessionSummary):
    """FR5-specific summary. This is a standard FR stim session with the
    possible addition of a recognition subtask at the end (only when not also a
    PS4 session).

    """
    recognized = Array(dtype=int, desc='item in recognition subtask recognized')
    rejected = Array(dtype=int, desc='lure item in recognition subtask rejected')

    def populate(self, events, recall_probs=None, is_ps4_session=False):
        FRStimSessionSummary.populate(self, events, recall_probs, is_ps4_session)
        self.recognized = events.recognized
