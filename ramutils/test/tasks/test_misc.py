import os
import glob
import pytest
import functools
import pandas as pd

from ramutils.reports.summary import FRSessionSummary, ClassifierSummary, MathSummary
from ramutils.tasks import build_static_report
from pkg_resources import resource_filename


datafile = functools.partial(resource_filename,
                             'ramutils.test.test_data')


def test_build_report_from_cached_results():
    session_summaries = [FRSessionSummary.from_hdf(datafile(
        'input/report_db/R1354E_FR1_1_session_summary.h5'))]

    classifier_summaries = [ClassifierSummary.from_hdf(datafile(
        'input/report_db/R1354E_FR1_1_classifier_summary.h5'))]

    math_summaries = [MathSummary.from_hdf(datafile(
        'input/report_db/R1354E_FR1_1_math_summary.h5'))]

    target_selection_table = pd.read_csv(datafile(
        'input/report_db/R1354E_FR1_1_delta_hfa_table.csv'))

    assert session_summaries is not None
    assert classifier_summaries is not None
    assert target_selection_table is not None

    report = build_static_report('R1354E', 'FR1', session_summaries,
                                 math_summaries, target_selection_table,
                                 classifier_summaries,
                                 datafile(
                                     'output/')).compute()

    assert report is not None
    output_files = glob.glob(datafile('output/*.html'))
    assert len(output_files) == 1

    # Clean up
    for f in output_files:
        os.remove(f)

    return

