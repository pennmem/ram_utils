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

    target_selection_table, classifier_summaries, session_summaries, \
    math_summaries, hmm_results = load_existing_results(
        'R1354E', 'FR1', [1], False, datafile('input/report_db'),
        datafile('')).compute()

    assert session_summaries is not None
    assert classifier_summaries is not None
    assert target_selection_table is not None

    report = build_static_report('R1354E', 'FR1', session_summaries,
                                 math_summaries, target_selection_table,
                                 classifier_summaries,
                                 datafile('output/')).compute()

    assert report is not None
    output_files = glob.glob(datafile('output/*.html'))
    assert len(output_files) == 1

    # Clean up
    for f in output_files:
        os.remove(f)

    # Check that non-existent data returns all None values
    target_selection_table, classifier_summaries, session_summaries, \
    math_summaries, hmm_results = load_existing_results(
        'R1345D', 'FR1', [1], False, datafile('input/report_db'),
        datafile('')).compute()

    results = [target_selection_table, classifier_summaries, session_summaries,
               math_summaries, hmm_results]

    assert all([r is None for r in results])
    return

