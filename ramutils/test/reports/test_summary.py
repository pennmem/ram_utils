import functools
import numpy as np
from pkg_resources import resource_filename
import warnings

import pytest

from numpy.testing import assert_equal, assert_almost_equal

from ptsa.data.readers import BaseEventReader
from traits.api import ListInt, ListFloat, ListBool

from ramutils.reports.summary import (
    Summary, SessionSummary, StimSessionSessionSummary, MathSummary,
    FRSessionSummary, FRStimSessionSummary
)

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
        # ignore UserWarnings from summary.populate calls
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


class TestStimSessionSummary:
    @pytest.mark.parametrize('is_ps4_session', [True, False])
    def test_populate(self, fr5_events, is_ps4_session):
        """Basic tests that data was populated correctly from events."""
        summary = StimSessionSessionSummary()
        summary.populate(fr5_events, is_ps4_session)
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

