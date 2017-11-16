from __future__ import division

from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import pytz

from traits.api import Array, ArrayOrNone

from ramutils.schema import Schema

__all__ = [
    'SessionSummary',
    'FRSessionSummary',
    'FRStimSessionSummary',
    'FR5SessionSummary',
]


class Summary(Schema):
    """Base class for all summary objects."""
    _events = ArrayOrNone(desc='all events from a session')
    phase = Array(desc='list phase type (stim, non-stim, etc.)')

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, new_events):
        if self._events is None:
            self._events = new_events


class SessionSummary(Summary):
    """Base class for single-session objects."""
    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, new_events):
        """Only allow setting of events which contain a single session."""
        assert len(np.unique(new_events['session'])) == 1, "events should only be from a single session"
        if self._events is None:
            self._events = new_events

    @property
    def session_number(self):
        """Returns the session number for this summary."""
        return self.events.session[0]

    @property
    def session_length(self):
        """Computes the total amount of time the session lasted in seconds."""
        start = self.events.mstime.min()
        end = self.events.mstime.max()
        return (end - start) / 1000.

    @property
    def session_datetime(self):
        """Returns a timezone-aware datetime object of the end time of the
        session in UTC.

        """
        timestamp = self.events.mstime.max() / 1000.
        return datetime.fromtimestamp(timestamp, pytz.utc)

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
            # these attributes won't be included
            ignore = [
                'events',  # we don't need events in the dataframe
                'rejected', 'region',  # FIXME: don't ignore
            ]

            # also ignore phase for events that predate it
            if 'phase' in self.visible_traits():
                if len(self.phase) == 0:
                    ignore += ['phase']

            columns = {
                trait: getattr(self, trait)
                for trait in self.visible_traits()
                if trait not in ignore
            }
            self._df = pd.DataFrame(columns)
        return self._df

    def populate(self, events):
        """Populate attributes and store events."""
        self.events = events
        try:
            self.phase = events.phase
        except AttributeError:
            warnings.warn("phase not found in events (probably pre-dates phase)",
                          UserWarning)


class MathSummary(SessionSummary):
    """Summarizes data from math distractor periods. Input events must either
    be all events (which include math events) or just math events.

    """
    @property
    def num_problems(self):
        """Returns the total number of problems solved by the subject."""
        return len(self.events[self.events.type == b'PROB'])

    @property
    def num_correct(self):
        """Returns the number of problems solved correctly."""
        return len(self.events[self.events.iscorrect == 1])

    @property
    def percent_correct(self):
        """Returns the percentage of problems solved correctly."""
        return 100 * self.num_correct / self.num_problems

    @property
    def problems_per_list(self):
        """Returns the mean number of problems per list."""
        n_lists = len(np.unique(self.events.list))
        return self.num_problems / n_lists

    @staticmethod
    def total_num_problems(summaries):
        """Get total number of problems for multiple sessions.

        Parameters
        ----------
        summaries : List[MathSummary]

        Returns
        -------
        : int

        """
        return sum(summary.num_problems for summary in summaries)

    @staticmethod
    def total_num_correct(summaries):
        """Get the total number of correctly answered problems for multiple
        sessions.

        Parameters
        ----------
        summaries : List[MathSummary]

        Returns
        -------
        : int

        """
        return sum(summary.num_correct for summary in summaries)

    @staticmethod
    def total_percent_correct(summaries):
        """Get the percent correct problems for multiple sessions.

        Parameters
        ----------
        summaries : List[MathSummary]

        Returns
        -------
        : float

        """
        probs = MathSummary.total_num_problems(summaries)
        correct = MathSummary.total_num_correct(summaries)
        return 100 * correct / probs

    @staticmethod
    def total_problems_per_list(summaries):
        """Get the mean number of problems per list for multiple sessions.

        FIXME: this doesn't seem to match R1111M's FR1 report.

        Parameters
        ----------
        summaries : List[MathSummary]

        Returns
        -------
        float

        """
        n_lists = sum(len(np.unique(summary.events[summary.events.list]))
                      for summary in summaries)
        return MathSummary.total_num_problems(summaries) / n_lists


class FRSessionSummary(SessionSummary):
    """Free recall session summary data."""
    item = Array(dtype='|U256', desc='list item (a.k.a. word)')
    session = Array(dtype=int, desc='session number')
    listno = Array(dtype=int, desc="item's list number")
    serialpos = Array(dtype=int, desc='item serial position')

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
        SessionSummary.populate(self, events)
        self.item = events.item_name
        self.session = events.session
        self.listno = events.list
        self.serialpos = events.serialpos
        self.recalled = events.recalled
        self.thresh = [0.5] * len(events)
        self.prob = recall_probs if recall_probs is not None else [-999] * len(events)

    @property
    def num_lists(self):
        """Returns the total number of lists."""
        return len(self.to_dataframe().listno.unique())

    @property
    def percent_recalled(self):
        """Calculates the percentage correctly recalled words."""
        # FIXME: is length of events always equal to number of items?
        return 100 * len(self.events[self.events.recalled == True]) / len(self.events)


# FIXME
# class CatFRSessionSummary(FRSessionSummary):
#     """Extends standard FR session summaries for categorized free recall
#     experiments.
#
#     """
#     irt_within_cat = Float(desc='average inter-response time within categories')
#     irt_between_cat = Float(desc='average inter-response time between categories')


class StimSessionSessionSummary(SessionSummary):
    """SessionSummary data specific to sessions with stimulation."""
    is_stim_list = Array(dtype=np.bool, desc='item is from a stim list')
    is_post_stim_item = Array(dtype=np.bool, desc='stimulation occurred on the previous item')
    is_stim_item = Array(dtype=np.bool, desc='stimulation occurred on this item')
    is_ps4_session = Array(dtype=np.bool, desc='list is part of a PS4 session')

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
        SessionSummary.populate(self, events)

        self.is_stim_list = [e.phase == 'STIM' for e in events]
        self.is_stim_item = events.is_stim
        self.is_post_stim_item = [False] + [events.is_stim[i - 1] for i in range(1, len(events))]
        self.is_ps4_session = [is_ps4_session] * len(events)

        # FIXME: region

        # FIXME: amplitudes, etc. should be CSV for multistim
        self.stim_anode_tag = [e.stim_params.anode_label for e in events]
        self.stim_cathode_tag = [e.stim_params.cathode_label for e in events]
        self.pulse_frequency = [e.stim_params.pulse_freq for e in events]
        self.amplitude = [e.stim_params.amplitude for e in events]
        self.duration = [e.stim_params.stim_duration for e in events]


class FRStimSessionSummary(FRSessionSummary, StimSessionSessionSummary):
    """SessionSummary for FR sessions with stim."""
    def populate(self, events, recall_probs=None, is_ps4_session=False):
        FRSessionSummary.populate(self, events, recall_probs)
        StimSessionSessionSummary.populate(self, events, is_ps4_session)

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
