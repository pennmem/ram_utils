import pytest
import functools
import pandas as pd

from pkg_resources import resource_filename
from ramutils.tasks.behavioral_analysis import estimate_effects_of_stim
from ramutils.reports.summary import FRStimSessionSummary


datafile = functools.partial(resource_filename,
                             'ramutils.test.test_data.input')


@pytest.mark.slow
def test_estimate_effects_of_stim():
    sample_df = pd.read_csv(datafile(
        "/summaries/sample_stim_session_summary.csv"))
    sample_summary = FRStimSessionSummary()
    sample_summary.populate_from_dataframe(sample_df)
    sample_summaries = [sample_summary]
    result_traces = estimate_effects_of_stim('R1374T', 'catFR5',
                                             sample_summaries).compute()
    assert len(result_traces) == 3
