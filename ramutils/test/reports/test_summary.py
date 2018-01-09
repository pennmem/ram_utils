import functools
import numpy as np
import pandas as pd
from pkg_resources import resource_filename
import warnings

import pytest

from numpy.testing import assert_equal, assert_almost_equal

from traits.api import ListInt, ListFloat, ListBool

from ramutils.reports.summary import *
from ramutils.tasks.events import build_ps_data
datafile = functools.partial(resource_filename, 'ramutils.test.test_data')


@pytest.fixture(scope='session')
def fr5_events():
    """FR5 events for R1345D."""
    filename = datafile('fr5-events.npz')
    events = np.load(filename)['events'].view(np.recarray)
    return events[events.session == 0]


@pytest.fixture(scope='session')
def math_events():
    """Math events for all FR1 sessions of R1111M."""
    filename = datafile('R1111M_math_events.npz')
    events = np.load(filename)['events'].view(np.recarray)
    return events


class TestSummary:
    def test_to_dataframe(self):
        class MySessionSummary(SessionSummary):
            bools = ListBool()
            ints = ListInt()
            floats = ListFloat()

        summary = MySessionSummary(
            bools=[True, True, True],
            ints=[1, 2, 3],
            floats=[1., 2., 3.],
            phase=['a', 'b', 'c']
        )

        df = summary.to_dataframe()

        assert all(df.bools == summary.bools)
        assert all(df.ints == summary.ints)
        assert all(df.floats == summary.floats)

    def test_session_length(self, fr5_events):
        summary = SessionSummary()
        summary.events = fr5_events
        assert np.floor(summary.session_length) == 2475

    def test_session_datetime(self, fr5_events):
        summary = SessionSummary()
        summary.events = fr5_events
        dt = summary.session_datetime
        assert dt.tzinfo is not None
        assert dt.year == 2017
        assert dt.month == 10
        assert dt.day == 9
        assert dt.hour == 18
        assert dt.minute == 8
        assert dt.second == 25
        assert dt.utcoffset().total_seconds() == 0

    def test_populate(self):
        with pytest.raises(NotImplementedError):
            summary = Summary()
            summary.populate(None)

    def test_create(self, fr5_events):
        summary = SessionSummary.create(fr5_events)
        assert_equal(summary.events, fr5_events)


class TestMathSummary:
    @classmethod
    def setup_class(cls):
        # ignore UserWarnings from summaries.populate calls
        warnings.filterwarnings('ignore', category=UserWarning)

    @classmethod
    def teardown_class(cls):
        warnings.resetwarnings()

    @staticmethod
    def all_summaries(events):
        summaries = []
        for session in np.unique(events.session):
            summary = MathSummary()
            summary.populate(events[events.session == session])
            summaries.append(summary)
        return summaries

    def test_num_problems(self, math_events):
        probs = 0
        for session in np.unique(math_events.session):
            events = math_events[math_events.session == session]
            summary = MathSummary()
            summary.populate(events)
            probs += summary.num_problems

        assert probs == 308

    def test_num_correct(self, math_events):
        correct = 0
        for session in np.unique(math_events.session):
            events = math_events[math_events.session == session]
            summary = MathSummary()
            summary.populate(events)
            correct += summary.num_correct

        assert correct == 268

    def test_percent_correct(self, math_events):
        percents = []
        for session in np.unique(math_events.session):
            events = math_events[math_events.session == session]
            summary = MathSummary()
            summary.populate(events)
            percents.append(summary.percent_correct)

        assert_almost_equal(percents, [94, 76, 90, 85], decimal=0)

    def test_problems_per_list(self, math_events):
        ppl = []
        for session in np.unique(math_events.session):
            events = math_events[math_events.session == session]
            summary = MathSummary()
            summary.populate(events)
            ppl.append(summary.problems_per_list)

        assert_almost_equal(ppl, [3.28, 3.47, 3, 4.24], decimal=2)

    def test_total_num_problems(self, math_events):
        summaries = self.all_summaries(math_events)
        assert MathSummary.total_num_problems(summaries) == 308

    def test_total_num_correct(self, math_events):
        summaries = self.all_summaries(math_events)
        assert MathSummary.total_num_correct(summaries) == 268

    def test_total_percent_correct(self, math_events):
        summaries = self.all_summaries(math_events)
        assert np.floor(MathSummary.total_percent_correct(summaries)) == 87

    def test_total_problems_per_list(self, math_events):
        summaries = self.all_summaries(math_events)

        # FIXME: the existing R1111M FR1 report says this should be 3.62
        assert_almost_equal([MathSummary.total_problems_per_list(summaries)],
                            [3.46], decimal=2)


class TestFRSessionSummary:
    @classmethod
    def setup_class(cls):
        cls.summary = FRSessionSummary()
        events = fr5_events()
        probs = np.random.random(len(events))
        cls.summary.populate(events, probs)

    def test_no_probs_given(self, fr5_events):
        summary = FRSessionSummary()
        summary.populate(fr5_events)
        assert all(summary.prob == -999)

    def test_num_lists(self):
        assert self.summary.num_lists == 25

    def test_percent_recalled(self):
        assert self.summary.percent_recalled == 16

    @pytest.mark.parametrize('first', [True, False])
    def test_serialpos_probabilities(self, first):
        if first:
            expected = [0.2, 0.12, 0.08, 0.08, 0.08, 0.0, 0.08, 0.04, 0.08, 0.0, 0.0, 0.04]
        else:
            expected = [0.2, 0.16, 0.08, 0.16, 0.16, 0.12, 0.28, 0.2, 0.08, 0.16, 0.24, 0.08]

        probs = FRSessionSummary.serialpos_probabilities([self.summary], first)
        assert_almost_equal(probs, expected, decimal=2)


class TestCatFRSessionSummary:
    @classmethod
    def setup_class(cls):
        cls.summary = CatFRSessionSummary()
        events = fr5_events()
        probs = np.random.random(len(events))
        cls.summary.populate(events, probs)

    def test_to_dataframe(self):
        df = self.summary.to_dataframe()
        assert len(df) == len(self.summary.events)


class TestStimSessionSummary:
    @pytest.mark.parametrize('is_ps4_session', [True, False])
    def test_populate(self, fr5_events, is_ps4_session):
        """Basic tests that data was populated correctly from events."""
        summary = StimSessionSummary()
        summary.populate(fr5_events, is_ps4_session=is_ps4_session)
        df = summary.to_dataframe()

        assert len(df[df.phase == 'BASELINE']) == 36
        assert len(df[df.phase == 'STIM']) == 192
        assert len(df[df.phase == 'NON-STIM']) == 72


class TestFRStimSessionSummary:
    @pytest.mark.skip
    def test_num_nonstim_lists(self, fr5_events):
        summary = FRStimSessionSummary()
        summary.populate(fr5_events)
        assert summary.num_nonstim_lists == 2


class TestClassifierSummary:
    @classmethod
    def setup_class(cls):
        cls.subject = 'TEST_SUBJECT'
        cls.experiment = 'TEST'
        cls.sessions = ['NA']
        cls.recalls = np.random.random_integers(0, 1, 100)
        cls.predicted_probabilities = np.random.normal(.5, .03, size=100)
        cls.permuation_aucs = np.random.normal(.5, .01, size=200)
        cls.summary = ClassifierSummary()

    def test_populate(self):
        summary = ClassifierSummary()
        summary.populate(self.subject, self.experiment,
                         self.sessions, self.recalls,
                         self.predicted_probabilities, self.permuation_aucs,
                         encoding_only=False)
        assert np.array_equal(self.recalls, summary.true_outcomes)
        assert np.array_equal(self.predicted_probabilities, summary.predicted_probabilities)
        assert np.array_equal(self.permuation_aucs, summary.permuted_auc_values)
        assert 'encoding_only' in summary.metadata
        assert summary.metadata['encoding_only'] is False

        return

    def test_auc(self):
        summary = ClassifierSummary()
        summary.populate(self.subject, self.experiment,
                         self.sessions, self.recalls,
                         self.predicted_probabilities,
                         self.permuation_aucs)
        return

    def test_pvalue(self):
        pass

    def test_false_positive_rate(self):
        pass

    def test_true_positive_rate(self):
        pass

    def test_thresholds(self):
        pass

    def test_median_classifier_output(self):
        pass

    def test_low_tercile_diff_from_mean(self):
        pass

    def test_high_tercile_diff_from_mean(self):
        pass


class TestStimSessionSummary:
    @classmethod
    def setup_class(cls):
        cls.sample_summary_table = pd.read_csv(datafile(
            "/input/summaries/sample_stim_session_summary.csv"))

    def test_from_dataframe(self):
        stim_summary = StimSessionSummary()
        stim_summary.populate_from_dataframe(self.sample_summary_table)
        assert len(stim_summary.events) == 300


class TestFRStimSessionSummary:
    @classmethod
    def setup_class(cls):
        cls.sample_summary_table = pd.read_csv(datafile(
            "/input/summaries/sample_stim_session_summary.csv"))
        cls.sample_events = cls.sample_summary_table.to_records(index=False)
        cls.sample_summary = FRStimSessionSummary()
        cls.sample_summary.populate(cls.sample_events)

    def test_num_nonstim_lists(self):
        assert self.sample_summary.num_nonstim_lists == 9

    def test_num_stim_lists(self):
        assert self.sample_summary.num_stim_lists == 16

    def test_lists(self):
        lists = self.sample_summary.lists()
        assert min(lists) == 1
        assert max(lists) == 25

        stim_lists = self.sample_summary.lists(stim=True)
        assert len(stim_lists) == 16

    def test_stim_events_by_list(self):
        stim_events_by_list = self.sample_summary.stim_events_by_list
        assert min(stim_events_by_list) == 0
        assert max(stim_events_by_list) == 9

    def test_prob_stim_by_serialpos(self):
        prob_stim_by_serialpos = self.sample_summary.prob_stim_by_serialpos
        assert min(prob_stim_by_serialpos) > .46
        assert max(prob_stim_by_serialpos) > .52
        return

    def test_recalls_by_list(self):
        stim_recalls_by_list = self.sample_summary.recalls_by_list(
            stim_items_only=True)
        assert sum(stim_recalls_by_list) == 17

        nonstim_recalls_by_list = self.sample_summary.recalls_by_list(
            stim_items_only=False)
        assert sum(nonstim_recalls_by_list) == 38

    def test_prob_first_recall_by_serialpos(self):
        prob_first_recall_nonstim = self.sample_summary.prob_first_recall_by_serialpos(stim=False)
        assert max(prob_first_recall_nonstim) < 0.57

        prob_first_recall_stim = self.sample_summary.prob_first_recall_by_serialpos(stim=True)
        assert max(prob_first_recall_stim) < 0.13

    def test_prob_recall_by_serialpos(self):
        recall_by_serialpos = self.sample_summary.prob_recall_by_serialpos(
            stim_items_only=False)
        assert max(recall_by_serialpos) > 0.29
        assert min(recall_by_serialpos) < 0.53

        recall_by_serialpos = self.sample_summary.prob_recall_by_serialpos(
            stim_items_only=True)
        assert max(recall_by_serialpos) > 0.66
        assert min(recall_by_serialpos) == 0

    def test_delta_recall(self):
        delta_recall_stim = self.sample_summary.delta_recall(
            post_stim_items=False)
        assert np.isclose(delta_recall_stim, 9.704164)

        delta_recall_post_stim = self.sample_summary.delta_recall(
            post_stim_items=True)
        assert np.isclose(delta_recall_post_stim, 5.953408)

    def test_stim_parameters(self):
        stim_params = self.sample_summary.stim_parameters
        assert len(stim_params) == 1

    def test_recall_test_results(self):
        test_results = self.sample_summary.recall_test_results
        # TODO: Manually check these values to ensure accuracy and add
        # assertions


@pytest.mark.rhino
class TestPSSessionSummary:
    @classmethod
    def setup_class(cls):
        # Events file is too large to store in repo, so build it from scratch
        cls.sample_events = build_ps_data('R1374T', 'catFR5', 'ps4_events',
                                          None, '/Volumes/RHINO/').compute()
        cls.sample_summary = PSSessionSummary()

    def test_populate(self):
        self.sample_summary.populate(self.sample_events)
        assert len(self.sample_summary.events) == 3068

    def test_to_dataframe(self):
        self.sample_summary.populate(self.sample_events)
        df = self.sample_summary.to_dataframe()
        assert len(df) == 3068

    def test_decision(self):
        self.sample_summary.populate(self.sample_events)
        decision = self.sample_summary.decision
        assert decision['best_amplitude'] == 0.998
        assert decision['best_location'] == 'LA7_LA8'
        assert np.isclose(decision['pval'], 0.00608, 1e-3)

    def test_location_summary(self):
        self.sample_summary.populate(self.sample_events)
        location_summaries = self.sample_summary.location_summary
        assert np.isclose(location_summaries['LA7_LA8'][
                              'best_delta_classifier'], 0.030218, 1e-3)
        assert np.isclose(location_summaries['LC6_LC7'][
                              'best_delta_classifier'], 0.01033, 1e-3)




