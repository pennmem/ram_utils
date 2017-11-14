import numpy as np
from pkg_resources import resource_filename
import pytest
from traits.api import ListInt, ListFloat, ListBool

from ramutils.reports.summary import (
    Summary, StimSessionSummary, FRSessionSummary, FRStimSessionSummary
)


@pytest.fixture(scope='session')
def fr5_events():
    """FR5 events for R1345D."""
    filename = resource_filename('ramutils.test.test_data', 'fr5-events.npz')
    return np.load(filename)['events'].view(np.recarray)


class TestSummary:
    def test_to_dataframe(self):
        class MySummary(Summary):
            bools = ListBool()
            ints = ListInt()
            floats = ListFloat()

        summary = MySummary(
            bools=[True, True, True],
            ints=[1, 2, 3],
            floats=[1., 2., 3.],
            phase=['a', 'b', 'c']
        )

        df = summary.to_dataframe()

        assert all(df.bools == summary.bools)
        assert all(df.ints == summary.ints)
        assert all(df.floats == summary.floats)


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


class TestStimSessionSummary:
    @pytest.mark.parametrize('is_ps4_session', [True, False])
    def test_populate(self, fr5_events, is_ps4_session):
        """Basic tests that data was populated correctly from events."""
        summary = StimSessionSummary()
        summary.populate(fr5_events, is_ps4_session)
        df = summary.to_dataframe()

        assert len(df[df.phase == 'BASELINE']) == 72
        assert len(df[df.phase == 'STIM']) == 384
        assert len(df[df.phase == 'NON-STIM']) == 144


class TestFRStimSessionSummary:
    @pytest.mark.skip
    def test_num_nonstim_lists(self, fr5_events):
        summary = FRStimSessionSummary()
        summary.populate(fr5_events)
        assert summary.num_nonstim_lists == 2

