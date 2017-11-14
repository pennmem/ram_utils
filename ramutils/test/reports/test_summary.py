import numpy as np
from pkg_resources import resource_filename
import pytest
from traits.api import ListInt, ListFloat, ListBool

from ramutils.reports.summary import (
    SessionSummary, StimSessionSessionSummary, FRSessionSessionSummary, FRStimSessionSummary
)


@pytest.fixture(scope='session')
def fr5_events():
    """FR5 events for R1345D."""
    filename = resource_filename('ramutils.test.test_data', 'fr5-events.npz')
    events = np.load(filename)['events'].view(np.recarray)
    return events[events.session == 0]


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


class TestFRSessionSummary:
    @classmethod
    def setup_class(cls):
        cls.summary = FRSessionSessionSummary()
        events = fr5_events()
        probs = np.random.random(len(events))
        cls.summary.populate(events, probs)

    def test_no_probs_given(self, fr5_events):
        summary = FRSessionSessionSummary()
        summary.populate(fr5_events)
        assert all(summary.prob == -999)

    def test_num_lists(self):
        assert self.summary.num_lists == 25

    def test_percent_recalled(self):
        assert self.summary.percent_recalled == 16


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

