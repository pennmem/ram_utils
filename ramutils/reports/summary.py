from __future__ import division

from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import pytz

from traits.api import Array, ArrayOrNone, Float, String, DictStrAny, Dict
from sklearn.metrics import roc_auc_score, roc_curve
from traitschema import Schema

__all__ = [
    'Summary',
    'ClassifierSummary',
    'SessionSummary',
    'StimSessionSummary',
    'FRSessionSummary',
    'CatFRSessionSummary',
    'FRStimSessionSummary',
    'FR5SessionSummary',
    'MathSummary'
]


class ClassifierSummary(Schema):
    """ Classifier Summary Object """
    _predicted_probabilities = ArrayOrNone(desc='predicted recall probabilities')
    _true_outcomes = ArrayOrNone(desc='actual results for recall vs. non-recall')
    _permuted_auc_values = ArrayOrNone(desc='permuted AUCs')
    _metadata = DictStrAny(desc='Dictionary containing additional metadata')

    subject = String(desc='subject')
    experiment = String(desc='experiment')
    sessions = Array(desc='sessions summarized by the object')
    recall_rate = Float(desc='overall recall rate')
    low_terc_recall_rate = Float(desc='recall rate when predicted probability of recall was in lowest tercile')
    mid_terc_recall_rate = Float(desc='recall reate when predicted probability of recall was in middle tercile')
    high_terc_recall_rate = Float(desc='recall rate when predicted probability of recall was in highest tercile')

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, new_metadata):
        self._metadata = new_metadata

    @property
    def predicted_probabilities(self):
        return self._predicted_probabilities

    @predicted_probabilities.setter
    def predicted_probabilities(self, new_predicted_probabilities):
        if self._predicted_probabilities is None:
            self._predicted_probabilities = new_predicted_probabilities

    @property
    def true_outcomes(self):
        return self._true_outcomes

    @true_outcomes.setter
    def true_outcomes(self, new_true_outcomes):
        if self._true_outcomes is None:
            self._true_outcomes = new_true_outcomes

    @property
    def permuted_auc_values(self):
        return self._permuted_auc_values

    @permuted_auc_values.setter
    def permuted_auc_values(self, new_permuted_auc_values):
        if self._permuted_auc_values is None:
            self._permuted_auc_values = new_permuted_auc_values

    @property
    def auc(self):
        auc = roc_auc_score(self.true_outcomes, self.predicted_probabilities)
        return auc

    @property
    def pvalue(self):
        pvalue = np.count_nonzero((self.permuted_auc_values >= self.auc)) / float(len(self.permuted_auc_values))
        return pvalue

    @property
    def false_positive_rate(self):
        fpr, _, _ = roc_curve(self.true_outcomes, self.predicted_probabilities)
        fpr = fpr.tolist()
        return fpr

    @property
    def true_positive_rate(self):
        _, tpr, _ = roc_curve(self.true_outcomes, self.predicted_probabilities)
        tpr = tpr.tolist()
        return tpr

    @property
    def thresholds(self):
        _, _, thresholds = roc_curve(self.true_outcomes, self.predicted_probabilities)
        thresholds = thresholds.tolist()
        return thresholds

    @property
    def median_classifier_output(self):
        return np.median(self.predicted_probabilities)

    @property
    def low_tercile_diff_from_mean(self):
        return 100.0 * (self.low_terc_recall_rate - self.recall_rate) / self.recall_rate

    @property
    def mid_tercile_diff_from_mean(self):
        return 100.0 * (self.mid_terc_recall_rate - self.recall_rate) / self.recall_rate

    @property
    def high_tercile_diff_from_mean(self):
        return 100.0 * (self.high_terc_recall_rate - self.recall_rate) / self.recall_rate

    def populate(self, subject, experiment, session, true_outcomes,
                 predicted_probabilities, permuted_auc_values, **kwargs):
        """ Populate classifier performance metrics

        Parameters
        ----------
        subject: string
            Subject identifier
        experiment: string
            Name of the experiment
        session: string
            Session number
        true_outcomes: array_like
            Boolean array for if a word was recalled or not
        predicted_probabilities: array_like
            Outputs from the trained classifier for each word event
        permuted_auc_values: array_like
            AUC values from performing a permutation test on classifier

        Keyword Arguments
        -----------------
        Any kwargs passed to populate will be stored in the metadata field of
        the classifier summary object
        """
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.true_outcomes = true_outcomes
        self.predicted_probabilities = predicted_probabilities
        self.permuted_auc_values = permuted_auc_values
        self.metadata = kwargs

        thresh_low = np.percentile(predicted_probabilities, 100.0 / 3.0)
        thresh_high = np.percentile(predicted_probabilities, 2.0 * 100.0 / 3.0)

        low_tercile_mask = (predicted_probabilities <= thresh_low)
        high_tercile_mask = (predicted_probabilities >= thresh_high)
        mid_tercile_mask = ~(low_tercile_mask | high_tercile_mask)

        self.low_terc_recall_rate = np.sum(true_outcomes[low_tercile_mask]) / float(np.sum(
            low_tercile_mask))
        self.mid_terc_recall_rate = np.sum(true_outcomes[mid_tercile_mask]) / float(np.sum(
            mid_tercile_mask))
        self.high_terc_recall_rate = np.sum(true_outcomes[high_tercile_mask]) / float(
            np.sum(high_tercile_mask))

        self.recall_rate = np.sum(true_outcomes) / float(true_outcomes.size)

        return


class Summary(Schema):
    """Base class for all summary objects."""
    _events = ArrayOrNone(desc='task events from a session')
    _raw_events = ArrayOrNone(desc='all events from a session')
    phase = Array(desc='list phase type (stim, non-stim, etc.)')

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, new_events):
        if self._events is None:
            self._events = new_events

    @property
    def raw_events(self):
        return self._raw_events

    @raw_events.setter
    def raw_events(self, new_events):
        if self._raw_events is None:
            self._raw_events = new_events

    def populate(self, events, raw_events=None):
        raise NotImplementedError

    @classmethod
    def create(cls, events, raw_events=None):
        """Create a new summary object from events.

        Parameters
        ----------
        events : np.recarray
        raw_events: np.recarray

        """
        instance = cls()
        instance.populate(events, raw_events=raw_events)
        return instance


class SessionSummary(Summary):
    """Base class for single-session objects."""
    #FIXME: Isn't this redundant?
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
                '_events',  # we don't need events in the dataframe
                '_raw_events',
                '_repetition_ratios',
                '_metadata',
                'irt_within_cat',
                'irt_between_cat',
                'rejected',
                'region',  # FIXME: don't ignore
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

    def populate(self, events, raw_events=None):
        """Populate attributes and store events."""
        self.events = events
        self.raw_events = raw_events
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

    def populate(self, events, raw_events=None, recall_probs=None):
        """Populate data from events.

        Parameters
        ----------
        events : np.recarray
        raw_events: np.recarray
        recall_probs : np.ndarray
            Predicted probabilities of recall per item. If not given, assumed
            there is no relevant classifier and values of -999 are used to
            indicate this.

        """
        SessionSummary.populate(self, events, raw_events=raw_events)
        self.item = events.item_name
        self.session = events.session
        self.listno = events.list
        self.serialpos = events.serialpos
        self.recalled = events.recalled
        self.thresh = [0.5] * len(events)
        self.prob = recall_probs if recall_probs is not None else [-999] * len(events)

    @property
    def num_words(self):
        """ Number of words in the session """
        return len(self.events[self.events.type == 'WORD'])

    @property
    def num_correct(self):
        """ Number of correctly-recalled words """
        return np.sum(self.events[self.events.type == 'WORD'].recalled)

    @property
    def num_prior_list_intrusions(self):
        """ Calculates the number of prior list intrusions """
        return np.sum((self.raw_events.intrusion > 0))

    @property
    def num_extra_list_intrusions(self):
        """ Calculates the number of extra-list intrusions """
        return np.sum((self.raw_events.intrusion == -1))

    @property
    def num_lists(self):
        """Returns the total number of lists."""
        return len(self.to_dataframe().listno.unique())

    @property
    def percent_recalled(self):
        """Calculates the percentage correctly recalled words."""
        # FIXME: is length of events always equal to number of items?
        return 100 * self.num_correct / self.num_words

    @staticmethod
    def serialpos_probabilities(summaries, first=False):
        """Computes the mean recall probability by word serial position.

        Parameters
        ----------
        summaries : List[Summary]
            Summaries of sessions.
        first : bool
            When True, return probabilities that each serial position is the
            first recalled word. Otherwise, return the probability of recall
            for each word by serial position.

        Returns
        -------
        List[float]

        """
        columns = ['serialpos', 'list', 'recalled', 'type']
        events = pd.concat([pd.DataFrame(s.events[columns]) for s in summaries])
        events = events[events.type == 'WORD']

        if first:
            firstpos = np.zeros(len(events.serialpos.unique()), dtype=np.float)
            for listno in events.list.unique():
                try:
                    nonzero = events[(events.list == listno) & (events.recalled == 1)].serialpos.iloc[0]
                except IndexError:  # no items recalled this list
                    continue
                thispos = np.zeros(firstpos.shape, firstpos.dtype)
                thispos[nonzero - 1] = 1
                firstpos += thispos
            return (firstpos / events.list.max()).tolist()
        else:
            group = events.groupby('serialpos')
            return group.recalled.mean().tolist()


class CatFRSessionSummary(FRSessionSummary):
    """
        Extends standard FR session summaries for categorized free recall
        experiments.
    """
    _repetition_ratios = Dict(desc='Repetition ratio by subject')

    irt_within_cat = Array(desc='average inter-response time within categories')
    irt_between_cat = Array(desc='average inter-response time between categories')

    def populate(self, events, raw_events=None, recall_probs=None,
                 repetition_ratio_dict={}):
        FRSessionSummary.populate(self, events, raw_events=raw_events,
                                  recall_probs=recall_probs)
        self.repetition_ratios = repetition_ratio_dict

        # Calculate between and within IRTs based on events
        catfr_events = events[events.experiment == 'catFR1']
        cat_recalled_events = catfr_events[catfr_events.recalled == 1]
        irt_within_cat = []
        irt_between_cat = []
        for session in np.unique(catfr_events.session):
            cat_sess_recalls = cat_recalled_events[cat_recalled_events.session == session]
            for list in np.unique(cat_sess_recalls.list):
                cat_sess_list_recalls = cat_sess_recalls[cat_sess_recalls.list == list]
                irts = np.diff(cat_sess_list_recalls.mstime)
                within = np.diff(cat_sess_list_recalls.category_num) == 0
                irt_within_cat.extend(irts[within])
                irt_between_cat.extend(irts[within == False])

        self.irt_within_cat = irt_within_cat
        self.irt_between_cat = irt_between_cat

    @property
    def repetition_ratios(self):
        return self._repetition_ratios

    @repetition_ratios.setter
    def repetition_ratios(self, new_repetition_ratios):
        self._repetition_ratios = new_repetition_ratios

    @property
    def irt_within_category(self):
        return self.irt_within_cat

    @property
    def irt_between_category(self):
        return self.irt_between_cat


class StimSessionSummary(SessionSummary):
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

    def populate(self, events, raw_events=None, is_ps4_session=False):
        """Populate stim data from events.

        Parameters
        ----------
        events : np.recarray
        raw_events: np.recarray
        is_ps4_session : bool
            Whether or not this experiment is also a PS4 session.

        """
        SessionSummary.populate(self, events, raw_events=raw_events)

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


class FRStimSessionSummary(FRSessionSummary, StimSessionSummary):
    """SessionSummary for FR sessions with stim."""
    def populate(self, events, raw_events=None, recall_probs=None,
                 is_ps4_session=False):
        FRSessionSummary.populate(self,
                                  events,
                                  raw_events=raw_events,
                                  recall_probs=recall_probs)
        StimSessionSummary.populate(self, events,
                                    raw_events=raw_events,
                                    is_ps4_session=is_ps4_session)

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

    def populate(self, events, raw_events=None, recall_probs=None,
                 is_ps4_session=False):
        FRStimSessionSummary.populate(self, events,
                                      raw_events=raw_events,
                                      recall_probs=recall_probs,
                                      is_ps4_session=is_ps4_session)
        self.recognized = events.recognized
