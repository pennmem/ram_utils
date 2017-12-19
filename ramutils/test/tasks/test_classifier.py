import pytest
import functools
import numpy as np

from ramutils.parameters import FRParameters, PALParameters
from ramutils.tasks.classifier import perform_cross_validation

from pkg_resources import resource_filename
from sklearn.externals import joblib


datafile = functools.partial(resource_filename,
                             'ramutils.test.test_data.input')


@pytest.mark.parametrize('subject, params', [
    ('R1350D', FRParameters),  # multi-session FR
    ('R1353N', PALParameters),  # pal
    ('R1354E', FRParameters)  # fr and catfr
])
def test_perform_cross_validation_regression(subject, params):
    # Note: These expected outputs will change if *any* of the inputs change,
    #  i.e. classifier, powers, events, or parameters
    expected_output = {
        'R1350D': 0.5442,
        'R1353N': 0.8948, # Joint report
        'R1354E': 0.6276,
    }
    params = params().to_dict()
    classifier = joblib.load(
        datafile('/classifiers/{}_trained_classifier.pkl'.format(subject)))
    powers = np.load(
        datafile('/powers/{}_normalized_powers.npy'.format(subject)))
    events = np.load(
        datafile('/events/{}_task_events.npy'.format(subject))).view(
        np.recarray)
    classifier_summary = perform_cross_validation(classifier, powers, events,
                                                  10, **params).compute()
    assert np.isclose(classifier_summary.auc, expected_output[subject],
                      rtol=1e-3)
    return


@pytest.mark.parametrize('subject, params', [
    ('R1350D', FRParameters),  # multi-session FR
    ('R1353N', PALParameters),  # pal
    ('R1354E', FRParameters)  # fr and catfr
])
def test_perform_lolo_cross_validation_regression(subject, params):
    # Note: These expected outputs will change if *any* of the inputs change,
    #  i.e. classifier, powers, events, or parameters
    expected_output = {
        'R1350D': 0.5859,
        'R1353N': 0.7399,
        'R1354E': 0.545,
    }
    params = params().to_dict()
    classifier = joblib.load(
        datafile('/classifiers/{}_trained_classifier.pkl'.format(subject)))
    powers = np.load(
        datafile('/powers/{}_normalized_powers.npy'.format(subject)))
    events = np.load(
        datafile('/events/{}_task_events.npy'.format(subject))).view(
        np.recarray)

    # Select just the first session so that lolo cross validation is used
    sessions = np.unique(events.session)
    test_session = sessions[0]
    sess_mask = (events.session == test_session)
    events = events[events.session == test_session]
    powers = powers[sess_mask, :]
    classifier_summary = perform_cross_validation(classifier, powers, events, 10, **params).compute()
    assert np.isclose(classifier_summary.auc, expected_output[subject],
                      rtol=1e-3)

    return
