import pytest
import functools
import numpy as np

from ramutils.parameters import FRParameters, PALParameters
from ramutils.tasks.classifier import summarize_classifier
from ramutils.utils import load_event_test_data

from pkg_resources import resource_filename
from sklearn.externals import joblib


datafile = functools.partial(resource_filename,
                             'ramutils.test.test_data.input')


@pytest.mark.slow
class TestCrossValidation:
    @pytest.mark.parametrize('subject, params', [
        ('R1350D', FRParameters),  # multi-session FR
        ('R1353N', PALParameters),  # pal
        ('R1354E', FRParameters)  # fr and catfr
    ])
    def test_perform_cross_validation_regression(self, subject, params,
                                                 rhino_root):
        # Note: These expected outputs will change if *any* of the inputs
        # change, i.e. classifier, powers, events, or parameters
        expected_output = {
            'R1350D': 0.5486,
            'R1353N': 0.8790,
            'R1354E': 0.6292,
        }
        params = params().to_dict()
        classifier = joblib.load(
            datafile('/classifiers/{}_trained_classifier.pkl'.format(subject)))
        powers = np.load(
            datafile('/powers/{}_normalized_powers.npy'.format(subject)))
        events = load_event_test_data(
            datafile('/events/{}_task_events.npy'.format(subject)), rhino_root)
        classifier_summary = summarize_classifier(classifier, powers,
                                                  events, 10,
                                                  tag='test',
                                                  **params).compute()
        assert np.isclose(classifier_summary.auc, expected_output[subject],
                          rtol=1e-3)
        return

    @pytest.mark.parametrize('subject, params', [
        ('R1350D', FRParameters),  # multi-session FR
        ('R1353N', PALParameters),  # pal
        ('R1354E', FRParameters)  # fr and catfr
    ])
    def test_perform_lolo_cross_validation_regression(self, subject, params,
                                                      rhino_root):
        # Note: These expected outputs will change if *any* of the inputs
        # change, i.e. classifier, powers, events, or parameters
        expected_output = {
            'R1350D': 0.6053,
            'R1353N': 0.7352,
            'R1354E': 0.5505,
        }
        params = params().to_dict()
        classifier = joblib.load(
            datafile('/classifiers/{}_trained_classifier.pkl'.format(subject)))
        powers = np.load(
            datafile('/powers/{}_normalized_powers.npy'.format(subject)))
        events = load_event_test_data(
            datafile('/events/{}_task_events.npy'.format(subject)), rhino_root)

        # Select just the first session so that lolo cross validation is used
        sessions = np.unique(events.session)
        test_session = sessions[0]
        sess_mask = (events.session == test_session)
        events = events[events.session == test_session]
        powers = powers[sess_mask, :]
        classifier_summary = summarize_classifier(classifier,
                                                  powers,
                                                  events, 10,
                                                  tag='test',
                                                  **params).compute()
        assert np.isclose(classifier_summary.auc, expected_output[subject],
                          rtol=1e-3)

        return
