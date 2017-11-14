import numpy as np
from pkg_resources import resource_filename
import pytest
from traits.api import ListInt, ListFloat, ListBool

from ramutils.reports.summary import (
    Summary, FRSessionSummary, FRStimSessionSummary
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
            floats=[1., 2., 3.]
        )

        df = summary.to_dataframe()

        assert all(df.bools == summary.bools)
        assert all(df.ints == summary.ints)
        assert all(df.floats == summary.floats)


class TestFRSessionSummary:
    @classmethod
    def setup_class(cls):
        cls.summary = FRSessionSummary()
        cls.summary.populate(fr5_events())

    def test_num_lists(self):
        assert self.summary.num_lists == 25


class TestFRStimSessionSummary:
    @pytest.mark.skip
    def test_num_nonstim_lists(self, fr5_events):
        summary = FRStimSessionSummary()
        summary.populate(fr5_events)
        assert summary.num_nonstim_lists == 2
