from __future__ import division
from __future__ import unicode_literals

from datetime import datetime

import json
import numpy as np
import pandas as pd
import pytz
import pickle
import base64
from collections import OrderedDict
import io

from ramutils.utils import safe_divide
from ramutils.events import extract_subject, extract_experiment_from_events, \
    extract_sessions
from ramutils.bayesian_optimization import choose_location
from ramutils.exc import TooManySessionsError
from ramutils.parameters import ExperimentParameters
from ramutils.powers import save_power_plot, save_eeg_by_channel_plot
from ramutils.utils import encode_file
from ramutils.montage import (generate_pairs_for_classifier, get_distances,
                              get_used_pair_mask)
from ramutils.thetamod import tmi
from traitschema import Schema
from traits.api import Array, ArrayOrNone, Float, Unicode, Bool, Bytes, CArray


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
    'TICLFRSessionSummary',
    'PSSessionSummary',
    'LocationSearchSessionSummary',
    'MathSummary'
]


class ClassifierSummary(Schema):
    """ Classifier Summary Object """
    _predicted_probabilities = ArrayOrNone(
        desc='predicted recall probabilities')
    _true_outcomes = ArrayOrNone(
        desc='actual results for recall vs. non-recall')
    _permuted_auc_values = ArrayOrNone(desc='permuted AUCs')

    _frequencies = ArrayOrNone(
        desc='Frequencies the classifier was trained on')
    _pairs = ArrayOrNone(desc='Bipolar pairs used to train the classifier')
    _features = ArrayOrNone(desc='Feature matrix used to train the classifier')
    _coef = ArrayOrNone(desc = 'Classifier coefficients')

    subject = Unicode(desc='subject')
    experiment = Unicode(desc='experiment')
    sessions = Array(desc='sessions summarized by the object')
    recall_rate = Float(desc='overall recall rate')
    tag = Unicode(desc='name of the classifier')
    reloaded = Bool(desc='classifier was reloaded from hard disk')
    low_terc_recall_rate = Float(
        desc='recall rate when predicted probability of recall was in lowest tercile')
    mid_terc_recall_rate = Float(
        desc='recall reate when predicted probability of recall was in middle tercile')
    high_terc_recall_rate = Float(
        desc='recall rate when predicted probability of recall was in highest tercile')

    @property
    def id(self):
        session_str = ".".join([str(sess)
                                for sess in np.unique(self.sessions)])
        return ":".join([self.subject, self.experiment, session_str])

    @property
    def predicted_probabilities(self):
        """ Classifier output for each word encoding event """
        return self._predicted_probabilities

    @predicted_probabilities.setter
    def predicted_probabilities(self, new_predicted_probabilities):
        if self._predicted_probabilities is None:
            self._predicted_probabilities = new_predicted_probabilities

    @property
    def true_outcomes(self):
        """ Behavioral response (recalled/not-recalled) to each word encoding event"""
        return self._true_outcomes

    @true_outcomes.setter
    def true_outcomes(self, new_true_outcomes):
        if self._true_outcomes is None:
            self._true_outcomes = new_true_outcomes

    @property
    def permuted_auc_values(self):
        """ Array of AUC values from performing permutation test """
        return self._permuted_auc_values

    @permuted_auc_values.setter
    def permuted_auc_values(self, new_permuted_auc_values):
        if self._permuted_auc_values is None:
            self._permuted_auc_values = new_permuted_auc_values

    @property
    def auc(self):
        """ Classifier AUC """
        auc = roc_auc_score(self.true_outcomes, self.predicted_probabilities)
        return auc

    @property
    def pvalue(self):
        """ p-value of classifier AUC based on permuted AUCs """
        pvalue = np.count_nonzero(
            (self.permuted_auc_values >= self.auc)) / float(len(self.permuted_auc_values))
        return pvalue

    @property
    def false_positive_rate(self):
        """ False positive rate used for AUC curve """
        fpr, _, _ = roc_curve(self.true_outcomes, self.predicted_probabilities)
        fpr = fpr.tolist()
        return fpr

    @property
    def true_positive_rate(self):
        """ True positive rate used for AUC curve"""
        _, tpr, _ = roc_curve(self.true_outcomes, self.predicted_probabilities)
        tpr = tpr.tolist()
        return tpr

    @property
    def thresholds(self):
        """ Thresholds used for AUC curve """
        _, _, thresholds = roc_curve(
            self.true_outcomes, self.predicted_probabilities)
        thresholds = thresholds.tolist()
        return thresholds

    @property
    def median_classifier_output(self):
        """ Median of the classifier outputs """
        return np.median(self.predicted_probabilities)

    @property
    def confidence_interval_median_classifier_output(self):
        """
            95% confidence interval for the median of the classifier output. Used as a sniff test for if something is
            amiss. Should be centered around 0.5
        """
        sorted_probs = sorted(self.predicted_probabilities)
        n = len(self.predicted_probabilities)
        low_idx = int(round((n / 2.0) - ((1.96 * n**.5) / 2.0)))
        high_idx = int(round(1 + (n / 2.0) + ((1.96 * n**.5) / 2.0)))
        low_val = sorted_probs[low_idx]
        high_val = sorted_probs[high_idx]
        return low_val, high_val

    @property
    def low_tercile_diff_from_mean(self):
        """ % change in recall rate from overall recall when classifier output was in lowest tercile """
        return 100.0 * (self.low_terc_recall_rate - self.recall_rate) / self.recall_rate

    @property
    def mid_tercile_diff_from_mean(self):
        """ % change in recall rate from overall recall when classifier output was in middle tercile """
        return 100.0 * (self.mid_terc_recall_rate - self.recall_rate) / self.recall_rate

    @property
    def high_tercile_diff_from_mean(self):
        """ % change in recall rate from overall recall when classifier output was in highest tercile """
        return 100.0 * (self.high_terc_recall_rate - self.recall_rate) / self.recall_rate

    @property
    def features(self):
        return self._features if self._features is not None else np.array([])

    @property
    def pairs(self):
        return self._pairs if self._pairs is not None else np.array([])

    @property
    def frequencies(self):
        return self._frequencies if self._frequencies is not None else np.array([])

    @property
    def classifier_activation(self):
        """
        Forward model of classifier activation from Haufe et. al. 2014
        """
        if self._features is None:
            return np.array([])
        return np.dot(np.cov(self._features, rowvar=False), self._coef.squeeze())

    @property
    def classifier_activation_2d(self):
        return self.classifier_activation.reshape(
            len(self.pairs), len(self.frequencies)
        )

    @property
    def classifier_activation_by_region(self):
        if len(self.classifier_activation):
            activation_df = pd.DataFrame(data=self.classifier_activation_2d,
                                         index=self.pairs['region'])
            mean_activation = activation_df.groupby(activation_df.index).mean()
            return mean_activation.values.T
        else:
            return np.array([])

    @property
    def regions(self):
        """ List of unique electrode regions """
        if len(self.pairs):
            return [str(x) for x in np.unique(self.pairs['region'])]
        else:
            return []

    def populate(self, subject, experiment, session, true_outcomes,
                 predicted_probabilities, permuted_auc_values,
                 frequencies, pairs, features, coefficients,
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
        frequencies: array_like
            Frequencies used to train the classifier
        pairs: pd.DataFrame
            Metadata for each bipolar pair recorded from
        features: np.ndarray
            Feature matrix used to train the classifier,
            of shape [len(predicted_probabilities) , (len(pairs) * len(frequencies)].
        coefficients : np.array
            Array of classifier weights
        tag: str
            Name given to the classifier, used to differentiate between
            multiple classifiers
        reloaded: bool
            Indicates whether the classifier is reloaded from hard disk,
            i.e. is the actually classifier used. If false, then the
            classifier was created from scratch

        """
        self.subject = subject
        self.experiment = experiment
        self.sessions = session
        self.true_outcomes = true_outcomes
        self.predicted_probabilities = predicted_probabilities
        self.permuted_auc_values = permuted_auc_values
        self.tag = tag
        self.reloaded = reloaded
        self._frequencies = frequencies
        self._pairs = pairs
        self._features = features
        self._coef = coefficients

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
        """ Populate the summary object with the given events """
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
        """ For Math events, returns original events after excluding practice lists """
        events = np.rec.array(self._events)
        return events[events.list > -1]

    @events.setter
    def events(self, new_events):
        if self._events is None:
            self._events = np.rec.array(new_events)

    @property
    def session_number(self):
        """ Session number """
        return np.unique(self.events.session)[0]

    @property
    def num_problems(self):
        """Returns the total number of problems solved by the subject."""
        return len(self.events[(self.events.type == 'PROB') |
                               (self.events.type == b'PROB')])

    @property
    def num_lists(self):
        """ Number of lists at least partially completed in the session """
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
    """Base class for all session summary objects """
    _events = ArrayOrNone(
        desc='task-related events excluding math distractor events')
    _raw_events = ArrayOrNone(
        desc='all event types including math distractor events')
    _bipolar_pairs = Unicode(desc='bipolar pairs in montage')
    _excluded_pairs = Unicode(desc='bipolar pairs not used for classification '
                              'due to artifact or stimulation')
    _normalized_powers = ArrayOrNone(desc="normalized powers for all events "
                                          "and recorded pairs")

    @property
    def events(self):
        """ Numpy recarray of task events, i.e. the events used to train a classifier """
        return np.rec.array(self._events)

    @events.setter
    def events(self, new_events):
        if self._events is None:
            self._events = np.rec.array(new_events)

    @property
    def raw_events(self):
        """ :class:`np.rec.array` of all events (math and task) from the session """
        if self._raw_events is None:
            return None
        return np.rec.array(self._raw_events)

    @raw_events.setter
    def raw_events(self, new_events):
        if self._raw_events is None and new_events is not None:
            self._raw_events = np.rec.array(new_events)

    def populate(self, events, bipolar_pairs, excluded_pairs,
                 normalized_powers, raw_events=None):
        """ Abstract method to be overriden by child classes """
        raise NotImplementedError

    @classmethod
    def create(cls, events, bipolar_pairs, excluded_pairs,
               normalized_powers, raw_events=None):
        """Create a new summary object from events

        Parameters
        ----------
        events : :class:`np.recarray`
        raw_events: :class:`np.recarray`
        bipolar_pairs: dict
            Dictionary containing data in bipolar pairs in a montage
        excluded_pairs: dict
            Dictionary containing data on pairs excluded from analysis
        normalized_powers: :class:`np.ndarray`
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
        """ Subject ID associated with the session """
        return extract_subject(self.events, add_localization=True)

    @property
    def experiment(self):
        """ Experiment name """
        experiments = extract_experiment_from_events(self.events)
        return experiments[0]

    @property
    def session_number(self):
        """ Session number """
        sessions = extract_sessions(self.events)
        if len(sessions) != 1:
            raise TooManySessionsError("Single session expected for session "
                                       "summary")

        session = str(sessions[0])
        return session

    @property
    def id(self):
        return ":".join([self.subject, self.experiment, self.session_number])

    @property
    def events(self):
        """ :class:`np.recarray` of events """
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
        """ Returns a dictionary of bipolar pairs to be excluded in classifier training """
        return json.loads(self._excluded_pairs)

    @excluded_pairs.setter
    def excluded_pairs(self, new_excluded_pairs):
        self._excluded_pairs = json.dumps(new_excluded_pairs)

    @property
    def n_pairs(self):
        """ Returns the number of bipolar pairs in the recording"""
        return len(self.bipolar_pairs[self.subject]['pairs'])

    @property
    def normalized_powers(self):
        """ Powers normalized to 0 mean and unit variance """
        return self._normalized_powers

    @normalized_powers.setter
    def normalized_powers(self, new_normalized_powers):
        self._normalized_powers = new_normalized_powers

    @property
    def normalized_powers_covariance(self):
        return np.cov(self._normalized_powers.T)

    @property
    def normalized_powers_plot(self):
        """
        Plots the matrix of normalized powers for the session
        to the specified filename or file-like object,
        and returns the plot as a base64-encoded string
        """
        plot_buffer = io.BytesIO()
        save_power_plot(self.normalized_powers,
                        self.session_number, plot_buffer)
        return encode_file(plot_buffer)

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
        """ Number of lists completed in the session """
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
        """ Recall events that were either extra-list or prior-list intrusions """
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
        events = pd.concat([pd.DataFrame(s.events[columns])
                            for s in summaries])
        events = events[events.type == 'WORD']

        if first:
            firstpos = np.zeros(len(events.serialpos.unique()), dtype=np.float)
            for listno in events.list.unique():
                try:
                    nonzero = events[(events.list == listno) & (
                        events.recalled == 1)].serialpos.iloc[0]
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
    _repetition_ratios = Unicode(desc='Repetition ratio by subject')
    irt_within_cat = Array(
        desc='average inter-response time within categories')
    irt_between_cat = Array(
        desc='average inter-response time between categories')

    def populate(self, events, bipolar_pairs, excluded_pairs,
                 normalized_powers, raw_events=None,
                 repetition_ratio_dict={}):
        """ Populates the CatFRSessionSummary object """
        FRSessionSummary.populate(self, events,
                                  bipolar_pairs, excluded_pairs,
                                  normalized_powers,
                                  raw_events=raw_events)

        self.repetition_ratios = repetition_ratio_dict

        # Calculate between and within IRTs based on the REC_WORD events as found in all_events.json
        # Exclude all intrusions so that a transition between an intrusion and a recall will not be
        # counted towards either within or between times.
        catfr_events = events[(events.experiment == 'catFR1') &
                              (events.type == 'REC_EVENT') &
                              (events.intrusion == 0) &
                              (events.recalled == 1)]  # recalled == 0 indicates a baseline recall event
        cat_recalled_events = catfr_events[(catfr_events.recalled == 1)]
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
        """
            Dictionary where keys are subject identifiers for subjects completing at least one CatFR session and
            values are the repetition ratio for that subject by list
        """
        mydict = json.loads(self._repetition_ratios)
        mydict = {k: np.array(v) for k, v in mydict.items()}
        return mydict

    @property
    def repetition_ratios(self):
        """
            Dictionary where keys are subject identifiers for subjects completing at least one CatFR session and
            values are the repetition ratio for that subject averaged over the session
        """
        return np.hstack([np.nanmean(v) for k, v in self.raw_repetition_ratios.items()])

    @repetition_ratios.setter
    def repetition_ratios(self, new_repetition_ratios):
        serializable_ratios = {k: v.tolist() for k, v in
                               new_repetition_ratios.items()}
        self._repetition_ratios = json.dumps(serializable_ratios)

    @property
    def irt_within_category(self):
        """ Within-category item response time """
        return self.irt_within_cat

    @property
    def irt_between_category(self):
        """ Between category item response time """
        return self.irt_between_cat

    @property
    def subject_ratio(self):
        """ Repetition ratio for the current subject """
        return np.nanmean(self.raw_repetition_ratios[self.subject])


class StimSessionSummary(SessionSummary):
    """SessionSummary data specific to sessions with stimulation."""
    _post_stim_prob_recall = CArray(dtype=np.float,
                                         desc='classifier output in post stim period',
                                    default=np.array([]))
    _model_metadata = Bytes(desc="traces for Bayesian multilevel models")
    _post_stim_eeg = ArrayOrNone(desc='raw post-stim EEG')
    _stim_tstats = CArray

    @property
    def post_stim_prob_recall(self):
        """ Classifier output in the post-stim period """
        return self._post_stim_prob_recall

    @post_stim_prob_recall.setter
    def post_stim_prob_recall(self, new_post_stim_prob_recall):
        if new_post_stim_prob_recall is not None:
            self._post_stim_prob_recall = new_post_stim_prob_recall.flatten().tolist()

    @property
    def model_metadata(self):
        metadata = pickle.loads(self._model_metadata)
        return metadata

    @model_metadata.setter
    def model_metadata(self, new_model_metadata):
        """ Save the dictionary of model traces such that it can be stored in HDF5 """
        # Use pickle to convert to byte string and then base64 encode/decode to remove
        # NULL characters that are not handled well by HDF5
        metadata = pickle.dumps(new_model_metadata)
        metadata = base64.b64encode(metadata)
        self._model_metadata = metadata

    def populate(self, events, bipolar_pairs, excluded_pairs,
                 normalized_powers, post_stim_prob_recall=None,
                 raw_events=None, model_metadata={}, post_stim_eeg=None,
                 stim_tstats=None):

        """ Populate stim data from events """
        SessionSummary.populate(self, events,
                                bipolar_pairs,
                                excluded_pairs,
                                normalized_powers,
                                raw_events=raw_events)
        if post_stim_prob_recall is not None:
            self.post_stim_prob_recall = post_stim_prob_recall
        if len(model_metadata)>0:
            self.model_metadata = model_metadata
        if post_stim_eeg is not None:
            self._post_stim_eeg = post_stim_eeg
        if stim_tstats is not None:
            self._stim_tstats = stim_tstats


    @classmethod
    def stim_tstats_by_condition(cls, session_summaries):
        good_tstats = [x for summary in session_summaries
                       for x in
                       summary.stim_tstats[summary.stim_pvals > 0.001]]
        bad_tstats = [x for summary in session_summaries
                      for x in summary.stim_tstats[summary.stim_pvals < 0.001]]
        return good_tstats, bad_tstats


    @property
    def post_stim_eeg_plot(self):
        if self._post_stim_eeg is None:
            return ''
        else:
            pairs = ['%s-\n%s' % (pair['label0'], pair['label1'])
                     for pair in generate_pairs_for_classifier(self.bipolar_pairs, [])
                     ]
            bipolar_pairs = pd.DataFrame.from_dict(
                self.bipolar_pairs[self.subject]['pairs']
            )
            bipolar_pairs = bipolar_pairs.T.sort_values(by=['channel_1','channel_2'])
            bipolar_pairs = bipolar_pairs.T.to_dict(into=OrderedDict)
            bipolar_pairs = OrderedDict({self.subject: {'pairs': bipolar_pairs}})
            used_pair_mask = get_used_pair_mask(bipolar_pairs,
                                                self.excluded_pairs)
            return [encode_file(save_eeg_by_channel_plot(pairs[i:i+1],
                                                        self._post_stim_eeg[i:i+1],
                                                        used_pair_mask[i:i+1]))
                    for i in range(len(pairs))]

    @property
    def subject(self):
        """ Subject ID associated with the session """
        return extract_subject(self.events, add_localization=False)


class FRStimSessionSummary(FRSessionSummary, StimSessionSummary):
    """ SessionSummary for FR sessions with stim """

    def populate(self, events, bipolar_pairs,
                 excluded_pairs, normalized_powers, post_stim_prob_recall=None,
                 raw_events=None, model_metadata={}, post_stim_eeg=None,
                 stim_tstats=None):
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
                                    raw_events=raw_events,
                                    model_metadata=model_metadata,
                                    post_stim_eeg=post_stim_eeg,
                                    stim_tstats=stim_tstats)

    @staticmethod
    def combine_sessions(summaries):
        """ Combine information from multiple stim sessions """
        all_summary_dfs = []
        for summary in summaries:
            df = summary.to_dataframe()
            all_summary_dfs.append(df)

        combined_df = pd.concat(all_summary_dfs)
        return combined_df

    @staticmethod
    def all_post_stim_prob_recall(summaries, phase=None):
        post_stim_prob_recall = [
            summary.post_stim_prob_recall for summary in summaries]
        post_stim_prob_recall = np.concatenate(post_stim_prob_recall).tolist()
        return post_stim_prob_recall

    @staticmethod
    def pre_stim_prob_recall(summaries, phase=None):
        """ Classifier output in the pre-stim period for items that were eventually stimulated """
        df = FRStimSessionSummary.combine_sessions(summaries)
        pre_stim_probs = df[df['is_stim_item'] ==
                            True].classifier_output.values.tolist()
        return pre_stim_probs

    @staticmethod
    def num_nonstim_lists(summaries):
        """Returns the number of non-stim lists."""
        df = FRStimSessionSummary.combine_sessions(summaries)
        count = 0
        for listno in df.list.unique():
            if not df[df.list == listno].is_stim_list.all():
                count += 1
        return count

    @staticmethod
    def num_stim_lists(summaries):
        """Returns the number of stim lists."""
        df = FRStimSessionSummary.combine_sessions(summaries)
        count = 0
        for listno in df.list.unique():
            if df[df.list == listno].is_stim_list.all():
                count += 1
        return count

    @staticmethod
    def stim_events_by_list(summaries):
        """ Array containing the number of stim events by list """
        df = FRStimSessionSummary.combine_sessions(summaries)
        n_stim_events = df.groupby('list').is_stim_item.sum().tolist()
        return n_stim_events

    @staticmethod
    def prob_stim_by_serialpos(summaries):
        """ Array containing the probability of stimulation (mean of the classifier output) by serial position """
        df = FRStimSessionSummary.combine_sessions(summaries)
        return df.groupby('serialpos').classifier_output.mean().tolist()

    @staticmethod
    def lists(summaries, stim=None):
        """ Get a list of either stim lists or non-stim lists """
        df = FRStimSessionSummary.combine_sessions(summaries)
        if stim is not None:
            lists = df[df.is_stim_list == stim].list.unique().tolist()
        else:
            lists = df.list.unique().tolist()
        return lists

    @property
    def stim_columns(self):
        """ Fields associated with stimulation parameters """
        return ['stimAnodeTag', 'stimCathodeTag', 'location', 'amplitude',
                'stim_duration', 'pulse_freq']

    @staticmethod
    def stim_params_by_list(summaries):
        """ Returns a dataframe of stimulation parameters used within each session/list """
        df = FRStimSessionSummary.combine_sessions(summaries)
        df = df.replace('nan', np.nan)
        stim_columns = ['stimAnodeTag', 'stimCathodeTag', 'location',
                        'amplitude', 'stim_duration', 'pulse_freq']
        non_stim_columns = [c for c in df.columns if c not in stim_columns]
        static_columns = [c for c in ['subject', 'experiment', 'session', 'list']
                          if c in df.columns]

        stim_param_by_list = (df[(stim_columns + static_columns)]
                              .drop_duplicates()
                              .dropna(how='all'))

        # This ensures that for any given list, the stim parameters used
        # during that list are populated. This makes calculating post stim
        # item behavioral responses easier
        df = df[non_stim_columns]
        df = df.merge(stim_param_by_list, on=['subject', 'experiment',
                                              'session', 'list'], how='left')
        return df

    @staticmethod
    def stim_parameters(summaries):
        """ Returns a list of unique stimulation parameters used during the experiment """
        df = FRStimSessionSummary.stim_params_by_list(summaries)
        return FRStimSessionSummary.aggregate_stim_params_over_list(df)

    @staticmethod
    def aggregate_stim_params_over_list(df):
        df['location'] = df['location'].replace(np.nan, '--')
        stim_columns = ['stimAnodeTag', 'stimCathodeTag', 'location',
                        'amplitude',
                        'stim_duration', 'pulse_freq']
        grouped = (df.groupby(by=(stim_columns + ['is_stim_list']))
                   .agg({'is_stim_item': 'sum',
                         'subject': 'count'})
                   .rename(columns={'is_stim_item': 'n_stimulations',
                                    'subject': 'n_trials'})
                   .reset_index())
        return list(grouped.T.to_dict().values())

    @staticmethod
    def recall_test_results(summaries, experiment):
        """
            Returns a dictionary containing the results of chi-squared tests for the behavioral effects of stimulation.
            Comparisons include stim lists vs. non-stim lists, stim items vs. low-biomarker non-stim items, and post-stim
            items vers. low-biomarker non-stim items. All comparisons are done for each unique set of stimulation parameters
        """
        df = FRStimSessionSummary.stim_params_by_list(summaries)

        if "PS5" not in experiment:
            df = df[df.list > 3]
        else:
            df = df[df.list > -1]

        results = []
        for name, group in df.groupby(['stimAnodeTag', 'stimCathodeTag',
                                       'amplitude', 'stim_duration',
                                       'pulse_freq']):
            parameters = "/".join([str(n) for n in name])

            # Stim lists vs. non-stim lists
            n_correct_stim_list_recalls = group[group.is_stim_list == True].recalled.sum(
            )
            n_correct_nonstim_list_recalls = df[df.is_stim_list == False].recalled.sum(
            )
            n_stim_list_words = group[group.is_stim_list ==
                                      True].recalled.count()
            n_nonstim_list_words = df[df.is_stim_list ==
                                      False].recalled.count()
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
            n_correct_stim_item_recalls = group[group.is_stim_item == True].recalled.sum(
            )
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
            n_correct_post_stim_item_recalls = group[group.is_post_stim_item == True].recalled.sum(
            )
            n_post_stim_items = group[group.is_post_stim_item ==
                                      True].recalled.count()

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

    @staticmethod
    def recalls_by_list(summaries, stim_list_only=False):
        """ Number of recalls by list. Optionally returns results for only stim lists """
        df = FRStimSessionSummary.combine_sessions(summaries)
        if stim_list_only:
            recalls_by_list = (
                df[df.is_stim_list == stim_list_only]
                .groupby('list')
                .recalled
                .sum()
                .astype(int)
                .tolist())
        else:
            recalls_by_list = (
                df.groupby('list')
                  .recalled
                  .sum()
                  .astype(int)
                  .tolist())

        return recalls_by_list

    @staticmethod
    def prob_first_recall_by_serialpos(summaries, stim=False):
        """ Probability of recalling a word first by serial position. Optionally returns results for only stim items """
        df = FRStimSessionSummary.combine_sessions(summaries)
        events = df[df.is_stim_item == stim]

        firstpos = np.zeros(
            ExperimentParameters().number_of_items, dtype=np.float)
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

    @staticmethod
    def prob_recall_by_serialpos(summaries, stim_items_only=False):
        """ Probability of recall by serial position. Optionally returns results for only stim items """
        df = FRStimSessionSummary.combine_sessions(summaries)
        group = df[df.is_stim_item == stim_items_only].groupby('serialpos')
        return group.recalled.mean().tolist()

    @staticmethod
    def delta_recall(summaries, post_stim_items=False):
        """
            %change in item recall for stimulated items versus non-stimulated low biomarker items. Optionally return
            the same comparison, but for post-stim items
        """
        df = FRStimSessionSummary.combine_sessions(summaries)
        nonstim_low_bio_recall = df[(df.classifier_output < df.thresh) &
                                    (df.is_stim_list == False)].recalled.mean()
        if post_stim_items:
            recall_stim = df[df.is_post_stim_item == True].recalled.mean()

        else:
            recall_stim = df[df.is_stim_item == True].recalled.mean()

        delta_recall = 100 * \
            ((recall_stim - nonstim_low_bio_recall) / df.recalled.mean())

        return delta_recall



class FR5SessionSummary(FRStimSessionSummary):
    """ FR5-specific summary """

    def populate(self, events, bipolar_pairs,
                 excluded_pairs, normalized_powers, post_stim_prob_recall=None,
                 raw_events=None, model_metadata={}):
        """ Constructor for the object """
        FRStimSessionSummary.populate(self, events,
                                      bipolar_pairs,
                                      excluded_pairs,
                                      normalized_powers,
                                      raw_events=raw_events,
                                      post_stim_prob_recall=post_stim_prob_recall,
                                      model_metadata=model_metadata)


class TICLFRSessionSummary(FRStimSessionSummary):

    biomarker_events = ArrayOrNone

    def populate(self, events, bipolar_pairs,
                 excluded_pairs, normalized_powers, post_stim_prob_recall=None,
                 raw_events=None, model_metadata={}, post_stim_eeg=None,
                 biomarker_events=None, stim_tstats=None):

        FRStimSessionSummary.populate(self, events, bipolar_pairs,
                                      excluded_pairs, normalized_powers,
                                      post_stim_prob_recall,
                                      raw_events, model_metadata, post_stim_eeg,
                                      stim_tstats=stim_tstats)
        self.biomarker_events = biomarker_events

    def nstims(self, task_phase):
        """
        Number of stim events within t
        :param task_phase:
        :return:
        """
        if self.raw_events is None:
            return  0

        return (self.raw_events[self.raw_events.type=='STIM_ON'
                ].phase == task_phase).sum()

    def classifier_output(self, phase, position):
        """

        :param phase: either "ENCODING", "DISTRACT", or "RETRIEVAL"
        :param position: either "pre" or "post"
        :return:
        """
        biomarker_events = self.biomarker_events[
            self.biomarker_events['biomarker_value'] >= 0
            ]

        in_phase = biomarker_events['phase'] == phase
        this_position = biomarker_events['position'] == position

        if position == 'post':
            return biomarker_events[in_phase & this_position]['biomarker_value']
        else: # Only want """real""" pre-stim events, i.e. ones with a matching
              # post-stim event
            ids = biomarker_events[in_phase & this_position]['id']
            has_match = np.in1d(ids,
                                biomarker_events[~this_position
                                ]['id'])
            return biomarker_events[
                (in_phase & this_position)
            ][has_match]['biomarker_value']

    @property
    def stim_tstats(self):
        return self._stim_tstats['stim_tstats']

    @property
    def stim_pvals(self):
        return self._stim_tstats['stim_pvals']

    @staticmethod
    def pre_stim_prob_recall(summaries, phase=None):
        if phase is None:
            phases = ['ENCODING', 'DISTRACT', 'RETRIEVAL']
        else:
            phases = [phase]

        return np.concatenate([
            summary.classifier_output(phase_, 'pre')
            for summary in summaries for phase_ in phases
        ]).tolist()

    @staticmethod
    def all_post_stim_prob_recall(summaries, phase=None):
        if phase is None:
            phases = ['ENCODING', 'DISTRACT', 'RETRIEVAL']
        else:
            phases = [phase]

        return np.concatenate([
                                  summary.classifier_output(phase_, 'post')
                                  for summary in summaries
                                  for phase_ in phases
                              ]).tolist()


class PSSessionSummary(SessionSummary):
    """ Parameter Search experiment summary """

    def populate(self, events, bipolar_pairs, excluded_pairs,
                 normalized_powers, raw_events=None):
        SessionSummary.populate(self, events, bipolar_pairs, excluded_pairs,
                                normalized_powers, raw_events=raw_events)
        return

    @property
    def decision(self):
        """ Return a dictionary containing decision information from the Bayesian optimization algorithm """
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
        """
            Return a dictionary whose keys are the locations stimulated in the experiment and values are a dictionary
            containing additional metadata about the results from stimulating at that location
        """
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
                        location_summary['sem'] = float(
                            loc_decision_info['sem'])
                        location_summary['snr'] = float(
                            loc_decision_info['snr'])
                location_summaries[loc_tag] = location_summary

        return location_summaries


class LocationSearchSessionSummary(StimSessionSummary):

    connectivity = Array
    pre_psd = Array
    post_psd = Array
    bad_events_mask = CArray
    bad_channels_mask = CArray
    _regressions = ArrayOrNone


    @property
    def distmat(self):
        return get_distances(self.bipolar_pairs)

    @property
    def stim_channel_idxs(self):
        return tmi.get_stim_channels(self.bipolar_pairs,self.events)

    @property
    def regressions(self):
        if self._regressions is None:
            self._regresssions = tmi.regress_distance(
                self._pre_psd,self._post_psd,
                self._connectivity, self.distmat,
                self.stim_channel_idxs)
        return self._regressions

    @property
    def tmi(self):
        return tmi.compute_tmi(self.regressions)

    @staticmethod
    def stim_params_by_list(summaries):
        stim_params_table = FRStimSessionSummary.stim_params_by_list(summaries)
        stim_channel_labels = [summary.bipolar_pairs[idx]['label']
                               for summary in summaries
                               for idx in summary.stim_channel_idx
                               ]
        tmi_list = [tmi_val['zscore'] for summary in summaries
                    for tmi_val in summary.tmi]
        for (stim_channel, tmi_val) in zip(stim_channel_labels, tmi_list):
            anode,cathode = stim_channel.split('-')
            stim_params_table.loc[(stim_params_table.stimAnodeLabel == anode) &
                                  (stim_params_table.stimCathodeLabel == cathode),
                                  'tmi'] = tmi_val

        return stim_params_table

    @staticmethod
    def stim_params(summaries):
        df = LocationSearchSessionSummary.stim_params_by_list(summaries)
        return FRStimSessionSummary.aggregate_stim_params_over_list(df)

    def populate(self,events, bipolar_pairs, excluded_pairs,
                 connectivity, pre_psd, post_psd, bad_events_mask, bad_channel_mask,
                 stim_tstats=None):
        StimSessionSummary.populate(self, events, bipolar_pairs, excluded_pairs,
                                    None, stim_tstats=stim_tstats)
        self.connectivity = connectivity
        self.post_psd = post_psd
        self.pre_psd = pre_psd
        self.bad_channels_mask = bad_channel_mask
        self.bad_events_mask = bad_events_mask
