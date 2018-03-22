import numpy as np
import pytest
import functools
import pandas as pd

from pkg_resources import resource_filename
from ramutils.tasks.behavioral_analysis import estimate_effects_of_stim
from ramutils.reports.summary import FRStimSessionSummary


datafile = functools.partial(resource_filename, 'ramutils.test.test_data.input')


@pytest.fixture(scope='session')
def bipolar_pairs():
    return {}


@pytest.fixture(scope='session')
def excluded_pairs():
    return {}


@pytest.fixture(scope='session')
def normalized_powers():
    return np.array([[1,2], [3,4]])


# Not really a rhino test, but it is so slow that we don't want it running
# all the time
@pytest.mark.rhino
@pytest.mark.slow
def test_estimate_effects_of_stim(bipolar_pairs, excluded_pairs,
                                  normalized_powers):
    sample_df = pd.read_csv(datafile(
        "/summaries/sample_stim_session_summary.csv"))
    sample_recarray = sample_df.to_records(index=False)
    sample_summary = FRStimSessionSummary()
    sample_summary.populate(sample_recarray, bipolar_pairs, excluded_pairs,
                            normalized_powers)
    sample_summaries = [sample_summary]
    result_traces = estimate_effects_of_stim('R1374T', 'catFR5',
                                             sample_summaries).compute()
    assert len(result_traces) == 3
