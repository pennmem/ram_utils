from traits.api import ListInt, ListFloat, ListBool
from ramutils.reports.summary import Summary


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
