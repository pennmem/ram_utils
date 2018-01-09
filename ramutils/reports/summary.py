from __future__ import division

from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import pytz

from ramutils.utils import safe_divide
from ramutils.events import extract_subject, extract_experiment_from_events, \
    extract_sessions
from ramutils.bayesian_optimization import choose_location

from traitschema import Schema
from traits.api import Array, ArrayOrNone, Float, String, DictStrAny, Dict

from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.proportion import proportions_chisquare



__all__ = [
    'Summary',
    'ClassifierSummary',
    'SessionSummary',
    'StimSessionSummary',
    'FRSessionSummary',
    'CatFRSessionSummary',
    'FRStimSessionSummary',
    'FR5SessionSummary',
    'PSSessionSummary',
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
    _events = ArrayOrNone(desc='task events')
    _raw_events = ArrayOrNone(desc='all events')
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
    @property
    def session(self):
        sessions = extract_sessions(self.events)
        return sessions[0]

    @property
    def subject(self):
        return extract_subject(self.events)

    @property
    def experiment(self):
        experiments = extract_experiment_from_events(self.events)
        return experiments[0]

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
                'post_stim_prob_recall'
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
        return len(self.events[self.events.type == 'PROB'])

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
    @safe_divide
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
        return np.hstack([np.nanmean(v) for k, v in
                          self._repetition_ratios.items()])

    @repetition_ratios.setter
    def repetition_ratios(self, new_repetition_ratios):
        self._repetition_ratios = new_repetition_ratios

    @property
    def irt_within_category(self):
        return self.irt_within_cat

    @property
    def irt_between_category(self):
        return self.irt_between_cat

    @property
    def subject_ratio(self):
        return np.nanmean(self._repetition_ratios[self.subject])


class StimSessionSummary(SessionSummary):
    """SessionSummary data specific to sessions with stimulation."""
    is_stim_list = Array(dtype=np.bool, desc='item is from a stim list')
    is_post_stim_item = Array(dtype=np.bool, desc='stimulation occurred on the previous item')
    is_stim_item = Array(dtype=np.bool, desc='stimulation occurred on this item')
    is_ps4_session = Array(dtype=np.bool, desc='list is part of a PS4 session')
    prob_recall = Array(dtype=np.float, desc='probability of recalling a word')
    post_stim_prob_recall = ArrayOrNone(dtype=np.float,
                                        desc='classifier output in post stim period')

    stim_anode_tag = Array(desc='stim anode label')
    stim_cathode_tag = Array(desc='stim cathode label')
    region = Array(desc='brain region of stim pair')
    pulse_frequency = Array(dtype=np.float64, desc='stim pulse frequency [Hz]')
    amplitude = Array(dtype=np.float64, desc='stim amplitude [mA]')
    duration = Array(dtype=np.float64, desc='stim duration [ms]')

    def populate_from_dataframe(self, df, post_stim_prob_recall=None,
                                raw_events=None, is_ps4_session=False):
        events = df.to_records(index=False)
        self.populate(events,
                      post_stim_prob_recall=post_stim_prob_recall,
                      raw_events=raw_events,
                      is_ps4_session=is_ps4_session)

    def populate(self, events, post_stim_prob_recall=None,
                 raw_events=None,
                 is_ps4_session=False):
        """ Populate stim data from events.

        Parameters
        ----------
        events : np.recarray
        post_stim_prob_recall: np.array
            Classifier outputs during post stim period
        raw_events: np.recarray
        is_ps4_session : bool
            Whether or not this experiment is also a PS4 session.

        """
        SessionSummary.populate(self, events, raw_events=raw_events)

        self.is_stim_list = events.stim_list
        self.is_stim_item = events.is_stim_item
        self.is_post_stim_item = events.is_post_stim_item
        self.is_ps4_session = [is_ps4_session] * len(events)
        self.prob_recall = events.classifier_output
        self.post_stim_prob_recall = post_stim_prob_recall

        self.region = events.location
        self.stim_anode_tag = events.stimAnodeTag
        self.stim_cathode_tag = events.stimCathodeTag
        self.pulse_frequency = events.pulse_freq
        self.amplitude = events.amplitude
        self.duration = events.stim_duration


class FRStimSessionSummary(FRSessionSummary, StimSessionSummary):
    """ SessionSummary for FR sessions with stim """
    def populate(self, events, post_stim_prob_recall=None, raw_events=None,
                 recall_probs=None, is_ps4_session=False):
        FRSessionSummary.populate(self,
                                  events,
                                  raw_events=raw_events,
                                  recall_probs=recall_probs)
        StimSessionSummary.populate(self, events,
                                    post_stim_prob_recall=post_stim_prob_recall,
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

    @property
    def stim_events_by_list(self):
        df = self.to_dataframe()
        n_stim_events = df.groupby('listno').is_stim_item.sum().tolist()
        return n_stim_events

    @property
    def prob_stim_by_serialpos(self):
        df = self.to_dataframe()
        return df.groupby('serialpos').prob_recall.mean().tolist()

    def lists(self, stim=None):
        df = self.to_dataframe()
        if stim is None:
            lists = df.listno.unique().tolist()

        else:
            lists = df[df.is_stim_item == stim].listno.unique().tolist()

        return lists

    @property
    def stim_parameters(self):
        df = self.to_dataframe()
        unique_stim_info = df[['stim_anode_tag', 'stim_cathode_tag',
                               'region', 'amplitude', 'duration',
                               'pulse_frequency']].drop_duplicates().dropna()
        return list(unique_stim_info.T.to_dict().values())

    @property
    def recall_test_results(self):
        df = self.to_dataframe()
        results = []

        # Stim lists vs. non-stim lists
        n_correct_stim_list_recalls = df[df.is_stim_list == True].recalled.sum()
        n_correct_nonstim_list_recalls = df[df.is_stim_list ==
                                            False].recalled.sum()
        n_stim_list_words = df[df.is_stim_list == True].recalled.count()
        n_nonstim_list_words = df[df.is_stim_list == False].recalled.count()
        tstat_list, pval_list, _ = proportions_chisquare([
            n_correct_stim_list_recalls, n_correct_nonstim_list_recalls],
            [n_stim_list_words, n_nonstim_list_words])

        results.append({"comparison": "Stim Lists vs. Non-stim Lists",
                        "stim": (n_correct_stim_list_recalls,
                                 n_stim_list_words),
                        "non-stim": (n_correct_nonstim_list_recalls, n_nonstim_list_words),
                        "t-stat": tstat_list,
                        "p-value": pval_list})

        # stim items vs. non-stim low biomarker items
        n_correct_stim_item_recalls = df[df.is_stim_item == True].recalled.sum()
        n_correct_nonstim_item_recalls = df[(df.is_stim_item == False) &
                                            (df.prob_recall < 0.5)].recalled.sum()

        n_stim_items = df[df.is_stim_item == True].recalled.count()
        n_nonstim_items = df[(df.is_stim_item == False) &
                             (df.prob_recall < 0.5)].recalled.count()

        tstat_list, pval_list, _ = proportions_chisquare(
            [n_correct_stim_item_recalls, n_correct_nonstim_item_recalls],
            [n_stim_items, n_nonstim_items])

        results.append({
            "comparison": "Stim Items vs. Low Biomarker Non-stim Items",
            "stim": (n_correct_stim_item_recalls, n_stim_items),
            "non-stim": (n_correct_nonstim_item_recalls, n_nonstim_items),
            "t-stat": tstat_list,
            "p-value": pval_list})

        # post stim items vs. non-stim low biomarker items
        n_correct_post_stim_item_recalls = df[df.is_post_stim_item ==
                                              True].recalled.sum()

        n_post_stim_items = df[df.is_post_stim_item == True].recalled.count()

        tstat_list, pval_list, _ = proportions_chisquare(
            [n_correct_post_stim_item_recalls, n_correct_nonstim_item_recalls],
            [n_post_stim_items, n_nonstim_items])

        results.append({
            "comparison": "Post-stim Items vs. Low Biomarker Non-stim Items",
            "stim": (n_correct_post_stim_item_recalls, n_post_stim_items),
            "non-stim": (n_correct_nonstim_item_recalls, n_nonstim_items),
            "t-stat": tstat_list,
            "p-value": pval_list})

        return results

    def recalls_by_list(self, stim_items_only=False):
        df = self.to_dataframe()
        recalls_by_list = df[df.is_stim_item == stim_items_only].groupby(
            'listno').recalled.sum().astype(int).tolist()
        return recalls_by_list

    def prob_first_recall_by_serialpos(self, stim=False):
        df = self.to_dataframe()
        events = df[df.is_stim_item == stim]

        firstpos = np.zeros(len(events.serialpos.unique()), dtype=np.float)
        for listno in events.listno.unique():
            try:
                nonzero = events[(events.listno == listno) &
                                 (events.recalled == 1)].serialpos.iloc[0]
            except IndexError:  # no items recalled this list
                continue
            thispos = np.zeros(firstpos.shape, firstpos.dtype)
            thispos[nonzero - 1] = 1
            firstpos += thispos
        return (firstpos / events.listno.max()).tolist()

    def prob_recall_by_serialpos(self, stim_items_only=False):
        df = self.to_dataframe()
        group = df[df.is_stim_item == stim_items_only].groupby('serialpos')
        return group.recalled.mean().tolist()

    def delta_recall(self, post_stim_items=False):
        df = self.to_dataframe()
        nonstim_low_bio_recall = df[(df.prob_recall < 0.5) &
                                    (df.is_stim_list == False)].recalled.mean()
        if post_stim_items:
            recall_stim = df[df.is_post_stim_item == True].recalled.mean()

        else:
            recall_stim = df[df.is_stim_item == True].recalled.mean()

        delta_recall = 100 * (recall_stim - nonstim_low_bio_recall)

        return delta_recall


class FR5SessionSummary(FRStimSessionSummary):
    """FR5-specific summary. This is a standard FR stim session with the
    possible addition of a recognition subtask at the end (only when not also a
    PS4 session).

    """
    recognized = Array(dtype=int, desc='item in recognition subtask recognized')
    rejected = Array(dtype=int, desc='lure item in recognition subtask rejected')

    def populate(self, events, post_stim_prob_recall=None, raw_events=None,
                 recall_probs=None, is_ps4_session=False):
        FRStimSessionSummary.populate(self, events,
                                      raw_events=raw_events,
                                      post_stim_prob_recall=post_stim_prob_recall,
                                      recall_probs=recall_probs,
                                      is_ps4_session=is_ps4_session)
        self.recognized = events.recognized


class PSSessionSummary(SessionSummary):
    """ Parameter Search experiment summary """

    def populate(self, events, **kwargs):
        """ Populate stim data from events.

        Parameters
        ----------
        events : np.recarray
        """
        SessionSummary.populate(self, events)
        return

    @property
    def decision(self):
        """ Return a dictionary containing decision information """
        decision_dict = {
            'sham_dc': '',
            'sham_sem': '',
            'best_location': '',
            'best_amplitdue': '',
            'pval': '',
            'tstat': '',
            'tie': '',
            'tstat_vs_sham': '',
            'pval_vs_sham': '',
            'loc1': {},
            'loc2': {},
        }

        # FIXME: Find a way to use self.dataframe() instead
        events_df = pd.DataFrame.from_records([e for e in self.events],
                                              columns=self.events.dtype.names)

        decision = events_df.loc[events_df.type == 'OPTIMIZATION_DECISION']
        # If a session completes with convergence, there will be an
        # optimization decision event at the end. Otherwise, we need to
        # manually calculate one
        if len(decision) > 0:
            decision_dict['sham_dc'] = decision.sham.delta_classifier[0]
            decision_dict['sham_sem'] = decision.sham.sem[0]
            decision_dict['best_location'] = decision.decision.best_location[0]
            decision_dict['best_amplitude'] = (
                decision.loc1 if decision.loc1.loc_name == decision_dict[
                    'best_location'] else decision.loc2).amplitude[0]
            decision_dict['pval'] = decision.decision.p_val[0]
            decision_dict['tstat'] = decision.decision.t_stat[0]
            decision_dict['tie'] = decision.decision.tie[0]
            decision_dict['tstat_vs_sham'] = decision.sham.t_stat[0]
            decision_dict['pval_vs_sham'] = decision.sham.p_val[0]
            decision_dict['loc1'] = decision.loc1
            decision_dict['loc2'] = decision.loc2

        else:
            opt_events = events_df.loc[events_df.type == 'OPTIMIZATION']
            # This should win an award for least-readable line of python code
            (locations, loc_datasets) = zip(*[('_'.join(name),
                                               table.loc[:, ['amplitude',
                                                             'delta_classifier']].values) for (name, table) in opt_events.groupby(('anode_label', 'cathode_label'))])

            # TODO: include sham delta classifiers when we need to reconstruct
            # results
            if len(locations) > 1:
                decision, loc_info = choose_location(loc_datasets[0],
                                                     locations[0],
                                                     loc_datasets[1],
                                                     locations[1],
                                                     np.array([(ld.min(),
                                                                ld.max()) for ld in loc_datasets]),
                                                     None)
            else:
                return decision_dict # no decision reached

            for i, k in enumerate(loc_info):
                loc_info[k]['amplitude'] = loc_info[k]['amplitude'] / 1000
                decision_dict['loc%s' % (i+1)] = loc_info[k]

            decision_dict['tie'] = decision['Tie']
            decision_dict['best_location'] = decision['best_location_name']
            decision_dict['best_amplitude'] = loc_info[
                 decision_dict['best_location']]['amplitude']
            decision_dict['pval'] = decision['p_val']
            decision_dict['tstat'] = decision['t_stat']

        return decision_dict

    @property
    def location_summary(self):
        location_summaries = {}
        # FIXME: Also update the dataframe creation here
        events_df = pd.DataFrame.from_records([e for e in self.events],
                                              columns=self.events.dtype.names)
        events_by_location = events_df.groupby(['anode_label', 'cathode_label'])

        for location, loc_events in events_by_location:
            location_summary = {
               'amplitude': {},
               'delta_classifier': {},
               'post_stim_biomarker': {},
               'post_stim_amplitude': {},
               'best_amplitude': '',
               'best_delta_classifier': '',
               'sem': '',
               'snr': ''
            }

            if location[0] and location[1]:
                loc_tag = '%s_%s' % (location[0], location[1])

                opt_events = (loc_events.loc[loc_events.type == 'OPTIMIZATION']
                                        .groupby('list_phase'))

                for i, (phase, phase_opt_events) in enumerate(opt_events):
                    post_stim_phase_events = loc_events.loc[
                        (events_df.list_phase == phase) &
                        (events_df.type == 'BIOMARKER') &
                        (events_df.position == 'POST')]

                    decision = self.decision
                    if decision['loc1']['loc_name'] == loc_tag:
                        loc_decision_info = decision['loc1']
                    else:
                        loc_decision_info = decision['loc2']

                    location_summary['amplitude'][phase] \
                        = (phase_opt_events.amplitude.values / 1000.).tolist()
                    location_summary['delta_classifier'][phase] = \
                        phase_opt_events.delta_classifier.values.tolist()
                    location_summary['post_stim_biomarker'][
                        phase] = post_stim_phase_events.biomarker_value.tolist()
                    location_summary['post_stim_amplitude'][phase] = \
                        (post_stim_phase_events.amplitude.values /
                         1000.).tolist()

                    if len(loc_decision_info) > 0:
                        location_summary['best_amplitude'] = float(
                            loc_decision_info['amplitude'])
                        location_summary['best_delta_classifier'] = float(
                            loc_decision_info['delta_classifier'])
                        location_summary['sem'] = float(loc_decision_info['sem'])
                        location_summary['snr'] = float(loc_decision_info['snr'])
                location_summaries[loc_tag] = location_summary

        return location_summaries
