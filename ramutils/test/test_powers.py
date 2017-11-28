import os
import pytest
import functools
import numpy as np

from pkg_resources import resource_filename

from ramutils.powers import reduce_powers, compute_single_session_powers, \
    compute_powers
from ramutils.tasks import compute_normalized_powers, memory
from ramutils.parameters import FRParameters, PALParameters

datafile = functools.partial(resource_filename,
                             'ramutils.test.test_data.input')
RHINO_AVAIL = os.environ.get('RHINO_AVAIL')


@pytest.mark.skipif(RHINO_AVAIL == 'False', reason='rhino')
def test_compute_single_session_powers():
    # Cases: EEG from monopolar, mixed mode, and bipolar. EEG with removed bad
    # data and without. Log powers true and false
    events = np.load(datafile('/events/R1348J_task_events.npy')).view(
        np.recarray)[:5]
    params = FRParameters().to_dict()
    powers, events = compute_single_session_powers(1,
                                                   events,
                                                   params['encoding_start_time'],
                                                   params['encoding_end_time'],
                                                   params['encoding_buf'],
                                                   params['freqs'],
                                                   params['log_powers'],
                                                   params['filt_order'],
                                                   params['width'],
                                                   True,
                                                   bipolar_pairs=None)
    assert powers.shape[0] == 5

    return


@pytest.mark.skipif(RHINO_AVAIL == 'False', reason='rhino')
@pytest.mark.parametrize("event_file", [
    'R1348J_task_events.npy',
    'R1350D_task_events.npy'
])
def test_compute_powers(event_file):
    # Cases: Bipolar_pairs == none and not none. Single session/multi.
    events_per_session = 5

    events = np.load(datafile('/events/' + event_file)).view(np.recarray)

    sessions = np.unique(events.session)
    n_sessions = len(sessions)
    rand_indices = np.array([])
    for session in sessions:
        sess_idx = np.where(np.isin(events.session, session))[0]
        sess_rand_indices = np.random.choice(sess_idx, events_per_session)
        rand_indices = np.concatenate([rand_indices, sess_rand_indices])
    rand_indices = rand_indices.flatten().astype(int)
    selected_events = events[rand_indices]
    assert len(selected_events) == events_per_session * n_sessions

    params = FRParameters().to_dict()
    powers, final_events = compute_powers(selected_events,
                                          params['encoding_start_time'],
                                          params['encoding_end_time'],
                                          params['encoding_buf'],
                                          params['freqs'],
                                          params['log_powers'],
                                          params['filt_order'],
                                          params['width'],
                                          bipolar_pairs=None)
    assert powers.shape[0] == events_per_session * n_sessions
    assert np.allclose(np.mean(powers, axis=0), 0)
    # Dividing by n -1 for the z-scoring, so we won't be that close to 1
    assert np.allclose(np.std(powers, axis=0), 1, .2)

    return


def test_reduce_powers():
    n_electrodes = 100
    n_frequencies = 8
    n_events = 500
    sample_pow_mat = np.random.rand(n_events, n_electrodes, n_frequencies)

    # Case 1: Select 0 electrodes
    no_electrodes_mask = np.zeros(100, dtype=bool)
    reduced_pow_mat = reduce_powers(sample_pow_mat, no_electrodes_mask,
                                    n_frequencies)
    assert reduced_pow_mat.shape == (n_events, 0 * n_frequencies)

    # Case 2: Select all electrodes
    all_electrodes_mask = np.ones(100, dtype=bool)
    reduced_pow_mat = reduce_powers(sample_pow_mat, all_electrodes_mask,
                                    n_frequencies)
    assert reduced_pow_mat.shape == (n_events, n_electrodes * n_frequencies)

    # Case 3: Select a susbset of electrodes
    some_electrodes_masks = [
        np.array(np.random.random_integers(0, 1, size=n_electrodes)).astype(
            bool) for
        _ in range(10)]
    expected_sizes = [np.sum(mask) for mask in some_electrodes_masks]
    reduced_power_matrices = [reduce_powers(sample_pow_mat,
                                            some_electrodes_masks[i],
                                            n_frequencies) for i in range(
        len(some_electrodes_masks))]
    assert all([reduced_power_matrices[i].shape == (n_events, expected_sizes[i]
                                                 * n_frequencies) for i in
                range(len(expected_sizes))])
    return


def test_normalize_powers_by_session():
    # Not sure how much there is to test here
    return


@pytest.mark.skipif(RHINO_AVAIL == 'False', reason='rhino')
@pytest.mark.skip(reason='slow')
@pytest.mark.parametrize("events, exp_powers, parameters", [
    ('R1353N_task_events.npy', 'R1353N_normalized_powers.npy', PALParameters),
    ('R1354E_task_events.npy', 'R1354E_normalized_powers.npy', FRParameters),
    ('R1350D_task_events.npy', 'R1350D_normalized_powers.npy', FRParameters),
])
def test_regression_compute_normalized_powers(events, exp_powers, parameters):
    # Cases: Same as event partitions since powers are calculated independently
    # for each partition
    parameters = parameters().to_dict()
    orig_powers = np.load(datafile('/powers/' + exp_powers))
    events = np.load(datafile('/events/' + events)).view(np.recarray)
    new_powers, updated_events = compute_normalized_powers(events,
                                                  **parameters).compute()
    assert np.allclose(orig_powers, new_powers)
    memory.clear(warn=False)  # Clean up if the assertion passes

    return

