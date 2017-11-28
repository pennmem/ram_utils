import pytest
import functools
import numpy as np

from pkg_resources import resource_filename
from ramutils.classifier.weighting import \
    determine_weighting_scheme_from_events, get_equal_weights, \
    get_fr_sample_weights, get_pal_sample_weights

datafile = functools.partial(resource_filename,
                             'ramutils.test.test_data.input.events')


@pytest.mark.parametrize("event_file, exp_scheme", [
    ('R1350D_task_events.npy', 'FR'),
    ('R1348J_task_events.npy', 'FR'),
    ('R1354E_task_events.npy', 'FR'),
    ('R1353N_task_events.npy', 'PAL')
])
def test_determine_weighting_scheme_from_events(event_file, exp_scheme):
    events = np.load(datafile(event_file)).view(np.recarray)
    scheme = determine_weighting_scheme_from_events(events)
    assert scheme == exp_scheme
    return


@pytest.mark.parametrize("event_file", [
    'R1350D_task_events.npy',
    'R1348J_task_events.npy',
    'R1353N_task_events.npy',
    'R1354E_task_events.npy'
])
def test_get_equal_weights(event_file):
    events = np.load(datafile(event_file)).view(np.recarray)
    weights = get_equal_weights(events)
    assert len(weights) == len(events)
    assert np.allclose(weights, 1)
    return


@pytest.mark.parametrize("event_file", [
    'R1350D_task_events.npy',
    'R1348J_task_events.npy',
    'R1354E_task_events.npy'
])
def test_get_fr_sample_weights(event_file):
    # TODO: This check should be more robust
    events = np.load(datafile(event_file)).view(np.recarray)
    weights = get_fr_sample_weights(events, 2.5)
    assert np.allclose(weights, 1) == False
    return


@pytest.mark.parametrize("event_file", [
    'R1353N_task_events.npy'
])
def test_get_pal_sample_weights(event_file):
    # TODO: This check should be more robust
    events = np.load(datafile(event_file)).view(np.recarray)
    weights = get_pal_sample_weights(events, 7.2, 1.93)
    assert np.allclose(weights, 1) == False
    return
