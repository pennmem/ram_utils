import pytest
import joblib
import functools
import numpy as np

from ramutils.parameters import FRParameters, PALParameters
from ramutils.classifier.cross_validation import *

from pkg_resources import resource_filename


datafile = functools.partial(resource_filename,
                             'ramutils.test.test_data.input')


def test_run_lolo_xval():
    return


def test_run_loso_xval():
    return


def test_permuted_lolo_AUCs():
    return


def test_permuted_loso_AUCs():
    return


@pytest.mark.parametrize('subject, params', [
    ('R1350D', FRParameters),  # multi-session FR
    ('R1353N', PALParameters),  # pal
    ('R1354E', FRParameters)  # fr and catfr
])
def test_perform_cross_validation_regression(subject, params):
    # Note: These expected outputs will change if *any* of the inputs change,
    #  i.e. classifier, powers, events, or parameters
    expected_output = {
        'R1350D': 0.519384220099,
        'R1353N': 0.563130027492,
        'R1354E': 0.48615333774,
    }
    # Do both single session and all sessions
    params = params().to_dict()
    classifier = joblib.load(
        datafile('/classifiers/{}_trained_classifier.pkl'.format(subject)))
    powers = np.load(
        datafile('/powers/{}_normalized_powers.npy'.format(subject)))
    events = np.load(
        datafile('/events/{}_task_events.npy'.format(subject))).view(
        np.recarray)
    xval_output = perform_cross_validation(classifier, powers, events, 10,
                                           **params)
    assert np.isclose(xval_output['all'], expected_output[subject])
    return



