import os
import glob
import pytest
import functools

from ramutils.tasks.misc import load_existing_results
from ramutils.tasks import build_static_report
from pkg_resources import resource_filename


datafile = functools.partial(resource_filename,
                             'ramutils.test.test_data')


@pytest.mark.rhino
def test_build_report_from_cached_results():

    prior_results = load_existing_results(
        'R1354E', 'FR1', [1], False, datafile('input/report_db'),
        datafile('')).compute()

    assert prior_results['session_summaries'] is not None
    assert prior_results['classifier_evaluation_results'] is not None
    assert prior_results['target_selection_table'] is not None

    report = build_static_report('R1354E', 'FR1',
                                 prior_results['session_summaries'],
                                 prior_results['math_summaries'],
                                 prior_results['target_selection_table'],
                                 prior_results['classifier_evaluation_results'],
                                 datafile('output/')).compute()

    assert report is not None
    output_files = glob.glob(datafile('output/*.html'))
    assert len(output_files) == 1

    # Clean up
    for f in output_files:
        os.remove(f)

    # Check that non-existent data returns all None values
    prior_results = load_existing_results(
        'R1345D', 'FR1', [1], False, datafile('input/report_db'),
        datafile('')).compute()

    results = [v for k,v in prior_results.items()]

    assert all([r is None for r in results])
    return
