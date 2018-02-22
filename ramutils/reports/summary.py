from __future__ import division

from datetime import datetime
import warnings

import json
import numpy as np
import pandas as pd
import pytz

from ramutils.utils import safe_divide
from ramutils.events import extract_subject, extract_experiment_from_events, \
    extract_sessions
from ramutils.bayesian_optimization import choose_location
from ramutils.exc import TooManySessionsError

from traitschema import Schema
from traits.api import Array, ArrayOrNone, Float, String, Bool

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

    subject = String(desc='subject')
    experiment = String(desc='experiment')
    sessions = Array(desc='sessions summarized by the object')
    recall_rate = Float(desc='overall recall rate')
    tag = String(desc='name of the classifier')
    reloaded = Bool(desc='classifier was reloaded from hard disk')
    low_terc_recall_rate = Float(desc='recall rate when predicted probability of recall was in lowest tercile')
    mid_terc_recall_rate = Float(desc='recall reate when predicted probability of recall was in middle tercile')
    high_terc_recall_rate = Float(desc='recall rate when predicted probability of recall was in highest tercile')

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
    def confidence_interval_median_classifier_output(self):
        sorted_probs = sorted(self.predicted_probabilities)
        n = len(self.predicted_probabilities)
        low_idx = int(round((n / 2.0) - ((1.96 * n**.5) / 2.0)))
        high_idx = int(round(1 + (n / 2.0) + ((1.96 * n**.5) / 2.0)))
        low_val = sorted_probs[low_idx]
        high_val = sorted_probs[high_idx]
        return low_val, high_val

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
                 predicted_probabilities, permuted_auc_values,
                 tag='', reloaded=False):
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
        tag: str
            Name given to the classifier, used to differentiate between
            multiple classifiers
        reloaded: bool
            Indicates whether the classifier is reloaded from hard disk,
            i.e. is the actually classifier used. If false, then the
            classifier was created from scratch

        Keyword Arguments
        -----------------
        Any kwargs passed to populate will be stored in the metadata field of
        the classifier summary object
        """
        self.subject = subject
        self.experiment = experiment
        self.sessions = session
        self.true_outcomes = true_outcomes
        self.predicted_probabilities = predicted_probabilities
        self.permuted_auc_values = permuted_auc_values
        self.tag = tag
        self.reloaded = reloaded

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


class MathSummary(Schema):
    """Summarizes data from math distractor periods. Input events must either
    be all events (which include math events) or just math events.

    """
    _events = ArrayOrNone(desc='Math distractor task events')

    def populate(self, events):
        self.events = events

    def to_dataframe(self, recreate=False):
        """Convert the summary to a :class:`pd.DataFrame` for easier
        manipulation. This amounts to converting the events to a dataframe

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
            self._df = pd.DataFrame.from_records(self.events)
        return self._df

    @property
    def events(self):
        """ For Math events, explicitly exclude practice lists """
        events = np.rec.array(self._events)
        return events[events.list > -1]

    @events.setter
    def events(self, new_events):
        if self._events is None:
            self._events = np.rec.array(new_events)

    @property
    def session_number(self):
        return np.unique(self.events.session)[0]

    @property
    def num_problems(self):
        """Returns the total number of problems solved by the subject."""
        return len(self.events[(self.events.type == 'PROB') |
                               (self.events.type == b'PROB')])

    @property
    def num_lists(self):
        return len(np.unique(self.events.list))

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
        return self.num_problems / self.num_lists

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
        return sum([summary.num_problems for summary in summaries])

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
        return sum([summary.num_correct for summary in summaries])

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
        n_lists = sum([summary.num_lists for summary in summaries])
        return MathSummary.total_num_problems(summaries) / n_lists


class Summary(Schema):
    """Base class for all summary objects."""
    _events = ArrayOrNone(desc='task-related events excluding math distractor events')
    _raw_events = ArrayOrNone(desc='all event types including math distractor events')
    _bipolar_pairs = String(desc='bipolar pairs in montage')
    _excluded_pairs = String(desc='bipolar pairs not used for classification '
                                  'due to artifact or stimulation')
    _normalized_powers = ArrayOrNone(desc="normalized powers for all events "
                                          "and recorded pairs")

    @property
    def events(self):
        return np.rec.array(self._events)

    @events.setter
    def events(self, new_events):
        if self._events is None:
            self._events = np.rec.array(new_events)

    @property
    def raw_events(self):
        if self._raw_events is None:
            return None
        return np.rec.array(self._raw_events)

    @raw_events.setter
    def raw_events(self, new_events):
        if self._raw_events is None and new_events is not None:
            self._raw_events = np.rec.array(new_events)

    def populate(self, events, bipolar_pairs, excluded_pairs,
                 normalized_powers, raw_events=None):
        raise NotImplementedError

    @classmethod
    def create(cls, events, bipolar_pairs, excluded_pairs,
               normalized_powers, raw_events=None):
        """Create a new summary object from events

        Parameters
        ----------
        events : np.recarray
        raw_events: np.recarray
        bipolar_pairs: dict
            Dictionary containing data in bipolar pairs in a montage
        excluded_pairs: dict
            Dictionary containing data on pairs excluded from analysis
        normalized_powers: np.ndarray
            2D array of normalzied powers of shape n_events x (
            n_frequencies * n_bipolar_pairs)

        """
        instance = cls()
        instance.populate(events,
                          bipolar_pairs,
                          excluded_pairs,
                          normalized_powers,
                          raw_events=raw_events)
        return instance


class SessionSummary(Summary):
    """Base class for single-session objects."""

    @property
    def subject(self):
        return extract_subject(self.events)

    @property
    def experiment(self):
        experiments = extract_experiment_from_events(self.events)
        return experiments[0]

    @property
    def session_number(self):
        sessions = extract_sessions(self.events)
        if len(sessions) != 1:
            raise TooManySessionsError("Single session expected for session "
                                       "summary")

        session = str(sessions[0])
        return session

    @property
    def events(self):
        return np.rec.array(self._events)

    @events.setter
    def events(self, new_events):
        """Only allow setting of events which contain a single session."""
        if self._events is None:
            self._events = np.rec.array(new_events)
            assert len(np.unique(new_events['session'])) == 1, \
                "events should only be from a single session"

    @property
    def bipolar_pairs(self):
        """ Returns a dictionary of bipolar pairs"""
        return json.loads(self._bipolar_pairs)

    @bipolar_pairs.setter
    def bipolar_pairs(self, new_bipolar_pairs):
        self._bipolar_pairs = json.dumps(new_bipolar_pairs)

    @property
    def excluded_pairs(self):
        """ Returns a dictionary of bipolar pairs"""
        return json.loads(self._excluded_pairs)

    @excluded_pairs.setter
    def excluded_pairs(self, new_excluded_pairs):
        self._excluded_pairs = json.dumps(new_excluded_pairs)

    @property
    def normalized_powers(self):
        return self._normalized_powers

    @normalized_powers.setter
    def normalized_powers(self, new_normalized_powers):
        self._normalized_powers = new_normalized_powers

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

    @property
    def num_lists(self):
        return len(np.unique(self.events.list))

    def to_dataframe(self, recreate=False):
        """Convert the summary to a :class:`pd.DataFrame` for easier
        manipulation. This amounts to converting the events to a dataframe

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
            self._df = pd.DataFrame.from_records(self.events)

        return self._df

    def populate(self, events, bipolar_pairs, excluded_pairs,
                 normalized_powers, raw_events=None):
        """Populate attributes and store events."""
        self.events = events
        self.raw_events = raw_events
        self.bipolar_pairs = bipolar_pairs
        self.excluded_pairs = excluded_pairs
        self.normalized_powers = normalized_powers


class FRSessionSummary(SessionSummary):
    """Free recall session summary data."""
    def populate(self, events, bipolar_pairs, excluded_pairs,
                 normalized_powers, raw_events=None):
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
        SessionSummary.populate(self, events, bipolar_pairs, excluded_pairs,
                                normalized_powers, raw_events=raw_events)

    @property
    def intrusion_events(self):
        intr_events = self.raw_events[(self.raw_events.type == 'REC_WORD') &
                                      (self.raw_events.intrusion != -999) &
                                      (self.raw_events.intrusion != 0)]
        return intr_events

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

        return np.sum((self.intrusion_events.intrusion > 0))

    @property
    def num_extra_list_intrusions(self):
        """ Calculates the number of extra-list intrusions """
        return np.sum((self.intrusion_events.intrusion == -1))

    @property
    def num_lists(self):
        """Returns the total number of lists."""
        return len(np.unique(self.events.list))

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
    _repetition_ratios = String(desc='Repetition ratio by subject')
    irt_within_cat = Array(desc='average inter-response time within categories')
    irt_between_cat = Array(desc='average inter-response time between categories')

    def populate(self, events, bipolar_pairs, excluded_pairs,
                 normalized_powers, raw_events=None,
                 repetition_ratio_dict={}):
        FRSessionSummary.populate(self, events,
                                  bipolar_pairs, excluded_pairs,
                                  normalized_powers,
                                  raw_events=raw_events)

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
    def raw_repetition_ratios(self):
        mydict = json.loads(self._repetition_ratios)
        mydict = {k: np.array(v) for k, v in mydict.items()}
        return mydict

    @property
    def repetition_ratios(self):
        return np.hstack([np.nanmean(v) for k, v in self.raw_repetition_ratios.items()])

    @repetition_ratios.setter
    def repetition_ratios(self, new_repetition_ratios):
        serializable_ratios = {k: v.tolist() for k, v in
                               new_repetition_ratios.items()}
        self._repetition_ratios = json.dumps(serializable_ratios)

    @property
    def irt_within_category(self):
        return self.irt_within_cat

    @property
    def irt_between_category(self):
        return self.irt_between_cat

    @property
    def subject_ratio(self):
        return np.nanmean(self.raw_repetition_ratios[self.subject])


class StimSessionSummary(SessionSummary):
    """SessionSummary data specific to sessions with stimulation."""
    _post_stim_prob_recall = ArrayOrNone(dtype=np.float,
                                         desc='classifier output in post stim period')

    @property
    def post_stim_prob_recall(self):
        return self._post_stim_prob_recall

    @post_stim_prob_recall.setter
    def post_stim_prob_recall(self, new_post_stim_prob_recall):
        if new_post_stim_prob_recall is not None:
            self._post_stim_prob_recall = new_post_stim_prob_recall.flatten().tolist()

    def populate(self, events, bipolar_pairs, excluded_pairs,
                 normalized_powers, post_stim_prob_recall=None,
                 raw_events=None):
        """ Populate stim data from events """
        SessionSummary.populate(self, events,
                                bipolar_pairs,
                                excluded_pairs,
                                normalized_powers,
                                raw_events=raw_events)
        self.post_stim_prob_recall = post_stim_prob_recall



class FRStimSessionSummary(FRSessionSummary, StimSessionSummary):
    """ SessionSummary for FR sessions with stim """
    def populate(self, events, bipolar_pairs,
                 excluded_pairs, normalized_powers, post_stim_prob_recall=None,
                 raw_events=None):
        FRSessionSummary.populate(self,
                                  events,
                                  bipolar_pairs,
                                  excluded_pairs,
                                  normalized_powers,
                                  raw_events=raw_events)
        StimSessionSummary.populate(self, events,
                                    bipolar_pairs,
                                    excluded_pairs,
                                    normalized_powers,
                                    post_stim_prob_recall=post_stim_prob_recall,
                                    raw_events=raw_events)

    @property
    def pre_stim_prob_recall(self):
        df = self.to_dataframe()
        pre_stim_probs = df[df.is_stim_item == True].classifier_output.values.tolist()
        return pre_stim_probs

    @property
    def num_nonstim_lists(self):
        """Returns the number of non-stim lists."""
        df = self.to_dataframe()
        count = 0
        for listno in df.list.unique():
            if not df[df.list == listno].is_stim_list.all():
                count += 1
        return count

    @property
    def num_stim_lists(self):
        """Returns the number of stim lists."""
        df = self.to_dataframe()
        count = 0
        for listno in df.list.unique():
            if df[df.list == listno].is_stim_list.all():
                count += 1
        return count

    @property
    def stim_events_by_list(self):
        df = self.to_dataframe()
        n_stim_events = df.groupby('list').is_stim_item.sum().tolist()
        return n_stim_events

    @property
    def prob_stim_by_serialpos(self):
        df = self.to_dataframe()
        return df.groupby('serialpos').classifier_output.mean().tolist()

    def lists(self, stim=None):
        """ Get a list of either stim lists or non-stim lists """
        df = self.to_dataframe()
        if stim is not None:
            lists = df[df.is_stim_list == stim].list.unique().tolist()
        else:
            lists = df.list.unique().tolist()
        return lists

    @property
    def stim_columns(self):
        return ['stimAnodeTag', 'stimCathodeTag', 'location', 'amplitude',
                'stim_duration', 'pulse_freq']

    @property
    def stim_params_by_list(self):
        df = self.to_dataframe()
        df = df.replace('nan', np.nan)
        stim_columns = self.stim_columns
        non_stim_columns = [c for c in df.columns if c not in stim_columns]

        stim_param_by_list = (df[(stim_columns + ['list'])]
                                .drop_duplicates()
                                .dropna(how='all'))

        # This ensures that for any given list, the stim parameters used
        # during that list are populated. This makes calculating post stim
        # item behavioral responses easier
        df = df[non_stim_columns]
        df = df.merge(stim_param_by_list, on='list', how='left')
        return df

    @property
    def stim_parameters(self):
        df = self.stim_params_by_list
        df['location'] = df['location'].replace(np.nan, '--')
        grouped = (df.groupby(by=(self.stim_columns + ['is_stim_list']))
                     .agg({'is_stim_item' : 'sum',
                           'subject': 'count'})
                     .rename(columns={'is_stim_item': 'n_stimulations',
                                      'subject': 'n_trials'})
                     .reset_index())

        return list(grouped.T.to_dict().values())

    @property
    def recall_test_results(self):
        df = self.stim_params_by_list

        if "PS5" not in self.experiment:
            df = df[df.list > 3]
        else:
            df = df[df.list > -1]

        results = []
        for name, group in df.groupby(['stimAnodeTag', 'stimCathodeTag',
                                       'amplitude', 'stim_duration',
                                       'pulse_freq']):
            parameters = "/".join([str(n) for n in name])

            # Stim lists vs. non-stim lists
            n_correct_stim_list_recalls = group[group.is_stim_list == True].recalled.sum()
            n_correct_nonstim_list_recalls = df[df.is_stim_list == False].recalled.sum()
            n_stim_list_words = group[group.is_stim_list == True].recalled.count()
            n_nonstim_list_words = df[df.is_stim_list == False].recalled.count()
            tstat_list, pval_list, _ = proportions_chisquare([
                n_correct_stim_list_recalls, n_correct_nonstim_list_recalls],
                [n_stim_list_words, n_nonstim_list_words])

            results.append({"parameters": parameters,
                            "comparison": "Stim Lists vs. Non-stim Lists",
                            "stim": (n_correct_stim_list_recalls,
                                     n_stim_list_words),
                            "non-stim": (n_correct_nonstim_list_recalls, n_nonstim_list_words),
                            "t-stat": tstat_list,
                            "p-value": pval_list})

            # stim items vs. non-stim low biomarker items
            n_correct_stim_item_recalls = group[group.is_stim_item == True].recalled.sum()
            n_correct_nonstim_item_recalls = df[(df.is_stim_item == False) &
                                                (df.classifier_output <
                                                 df.thresh)].recalled.sum()

            n_stim_items = group[group.is_stim_item == True].recalled.count()
            n_nonstim_items = df[(df.is_stim_item == False) &
                                 (df.classifier_output <
                                  df.thresh)].recalled.count()

            tstat_list, pval_list, _ = proportions_chisquare(
                [n_correct_stim_item_recalls, n_correct_nonstim_item_recalls],
                [n_stim_items, n_nonstim_items])

            results.append({
                "parameters": parameters,
                "comparison": "Stim Items vs. Low Biomarker Non-stim Items",
                "stim": (n_correct_stim_item_recalls, n_stim_items),
                "non-stim": (n_correct_nonstim_item_recalls, n_nonstim_items),
                "t-stat": tstat_list,
                "p-value": pval_list})

            # post stim items vs. non-stim low biomarker items
            n_correct_post_stim_item_recalls = group[group.is_post_stim_item == True].recalled.sum()
            n_post_stim_items = group[group.is_post_stim_item == True].recalled.count()

            tstat_list, pval_list, _ = proportions_chisquare(
            [n_correct_post_stim_item_recalls, n_correct_nonstim_item_recalls],
            [n_post_stim_items, n_nonstim_items])

            results.append({
                "parameters": parameters,
                "comparison": "Post-stim Items vs. Low Biomarker Non-stim Items",
                "stim": (n_correct_post_stim_item_recalls, n_post_stim_items),
                "non-stim": (n_correct_nonstim_item_recalls, n_nonstim_items),
                "t-stat": tstat_list,
                "p-value": pval_list})

        return results

    def recalls_by_list(self, stim_list_only=False):
        df = self.to_dataframe()
        recalls_by_list = (
            df[df.is_stim_list == stim_list_only]
            .groupby('list')
            .recalled
            .sum()
            .astype(int)
            .tolist())
        return recalls_by_list

    def prob_first_recall_by_serialpos(self, stim=False):
        df = self.to_dataframe()
        events = df[df.is_stim_item == stim]

        firstpos = np.zeros(12, dtype=np.float)
        for listno in events.list.unique():
            try:
                nonzero = events[(events.list == listno) &
                                 (events.recalled == 1)].serialpos.iloc[0]
            except IndexError:  # no items recalled this list
                continue
            thispos = np.zeros(firstpos.shape, firstpos.dtype)
            thispos[nonzero - 1] = 1
            firstpos += thispos
        return (firstpos / events.list.max()).tolist()

    def prob_recall_by_serialpos(self, stim_items_only=False):
        df = self.to_dataframe()
        group = df[df.is_stim_item == stim_items_only].groupby('serialpos')
        return group.recalled.mean().tolist()

    def delta_recall(self, post_stim_items=False):
        df = self.to_dataframe()
        nonstim_low_bio_recall = df[(df.classifier_output < df.thresh) &
                                    (df.is_stim_list == False)].recalled.mean()
        if post_stim_items:
            recall_stim = df[df.is_post_stim_item == True].recalled.mean()

        else:
            recall_stim = df[df.is_stim_item == True].recalled.mean()

        delta_recall = 100 * (recall_stim - nonstim_low_bio_recall)

        return delta_recall


class FR5SessionSummary(FRStimSessionSummary):
    """ FR5-specific summary """

    def populate(self, events, bipolar_pairs,
                 excluded_pairs, normalized_powers, post_stim_prob_recall=None,
                 raw_events=None):
        FRStimSessionSummary.populate(self, events,
                                      bipolar_pairs,
                                      excluded_pairs,
                                      normalized_powers,
                                      raw_events=raw_events,
                                      post_stim_prob_recall=post_stim_prob_recall)


class PSSessionSummary(SessionSummary):
    """ Parameter Search experiment summary """

    def populate(self, events, bipolar_pairs, excluded_pairs,
                 normalized_powers, raw_events=None):
        SessionSummary.populate(self, events, bipolar_pairs, excluded_pairs,
                                normalized_powers, raw_events=raw_events)
        return

    @property
    def decision(self):
        """ Return a dictionary containing decision information """
        decision_dict = {
            'converged': True,
            'sham_dc': '',
            'sham_sem': '',
            'best_location': '',
            'best_amplitude': '',
            'pval': '',
            'tstat': '',
            'tie': '',
            'tstat_vs_sham': '',
            'pval_vs_sham': '',
            'loc1': {},
            'loc2': {},
        }
        events_df = pd.DataFrame.from_records([e for e in self.events],
                                              columns=self.events.dtype.names)

        decision = self.events[(self.events.type == 'OPTIMIZATION_DECISION')]
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
            decision_dict['converged'] = False
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
                return

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
        events_df = pd.DataFrame.from_records([e for e in self.events],
                                              columns=self.events.dtype.names)
        events_by_location = events_df.groupby(['anode_label',
                                                'cathode_label'])
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
