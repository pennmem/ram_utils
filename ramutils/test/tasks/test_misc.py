import os
import pytest
import functools
import pandas as pd

from ramutils.tasks.misc import save_all_output
from pkg_resources import resource_filename
from sklearn.externals import joblib


datafile = functools.partial(resource_filename,
                             'ramutils.test.test_data')


def test_save_all_output():
    session_summaries = joblib.load(datafile(
        '/input/report_db/R1354E_FR1_1_session_summaries.pkl'))

    math_summaries = joblib.load(datafile(
        '/input/report_db/R1354E_FR1_1_math_summaries.pkl'))

    sample_hfa_table = datafile('/input/powers/R1354E_hfa_ttest_table.csv')
    test_hfa_table = pd.read_csv(sample_hfa_table)

    classifier_summaries = joblib.load(datafile(
        '/input/report_db/R1354E_FR1_1_classifier_summaries.pkl'))

    success = save_all_output('TEST', 'FR1', session_summaries, math_summaries,
                              test_hfa_table, classifier_summaries,
                              datafile('/output/')).compute()

    assert os.path.exists(datafile('/output/TEST_FR1_all_delta_hfa_table.csv'))
    assert os.path.exists(datafile('/output/TEST_FR1_1_session_summary.h5'))
    assert os.path.exists(datafile('/output/TEST_FR1_1_math_summary.h5'))
    assert os.path.exists(datafile('/output/TEST_FR1_1_classifier_summary.h5'))
    return


def test_load_cached_results():
    return
