import pytest
import functools
import numpy as np
import pandas as pd

from pkg_resources import resource_filename

from ramutils.powers import reduce_powers, compute_single_session_powers, \
    compute_powers, reshape_powers_to_2d, reshape_powers_to_3d, calculate_delta_hfa_table
from ramutils.tasks import compute_normalized_powers, memory
from ramutils.parameters import FRParameters, PALParameters

datafile = functools.partial(resource_filename,
                             'ramutils.test.test_data.input')


@pytest.mark.rhino
def test_compute_single_session_powers():
    # Cases: EEG from monopolar, mixed mode, and bipolar. EEG with removed bad
    # data and without. Log powers true and false
    events = np.rec.array(np.load(datafile(
        '/events/R1348J_task_events_rhino.npy')))[:5]
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


@pytest.mark.rhino
@pytest.mark.parametrize("event_file", [
    'R1348J_task_events_rhino.npy',
    'R1350D_task_events_rhino.npy'
])
def test_compute_powers(event_file):
    # Cases: Bipolar_pairs == none and not none. Single session/multi.
    events_per_session = 5

    events = np.rec.array(np.load(datafile('/events/' + event_file)))

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

    # Case 1: Select 0 electrode
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



def test_reshape_powers_to_3d():
    test_powers = np.random.random(size=(10, 150))
    reshaped_powers = reshape_powers_to_3d(test_powers, 10)
    assert reshaped_powers.shape == (10, 15, 10)
    return


def test_reshape_powers_to_2d():
    test_powers = np.random.random(size=(10, 15, 10))
    reshaped_powers = reshape_powers_to_2d(test_powers)
    assert reshaped_powers.shape == (10, 150)
    return


@pytest.mark.parametrize("events, powers, exp_table, parameters", [
    ('R1354E_task_events_rhino.npy', 'R1354E_normalized_powers.npy', 'R1354E_hfa_ttest_table.csv', FRParameters)
])
def test_calculate_delta_hfa_table_regression(events, powers, exp_table, parameters):
    parameters = parameters().to_dict()
    powers = np.load(datafile('/powers/' + powers))
    events = np.rec.array(np.load(datafile('/events/' + events)))
    config_pairs = pd.read_csv(datafile('/montage/R1354E_montage_metadata.csv'), index_col=0)
    hfa_table = calculate_delta_hfa_table(config_pairs, powers, events, parameters['freqs'])
    old_hfa_table = pd.read_csv(datafile('/powers/' + exp_table))

    assert np.allclose(old_hfa_table['t_stat'].values, hfa_table['t_stat'].values)
    assert np.allclose(old_hfa_table['p_value'].values, hfa_table['p_value'].values)

    return


@pytest.mark.rhino
@pytest.mark.slow
@pytest.mark.parametrize("events, exp_powers, parameters", [
    # ('R1353N_task_events_rhino.npy', 'R1353N_normalized_powers.npy', PALParameters),
    ('R1354E_task_events_rhino.npy', 'R1354E_normalized_powers.npy', FRParameters),
    ('R1350D_task_events_rhino.npy', 'R1350D_normalized_powers.npy', FRParameters),
])
def test_regression_compute_normalized_powers(events, exp_powers, parameters):
    parameters = parameters().to_dict()
    events = np.rec.array(np.load(datafile('/events/' + events)))
    new_powers, updated_events = compute_normalized_powers(events,
                                                           **parameters).compute()

    orig_powers = np.load(datafile('/powers/' + exp_powers))
    assert np.allclose(orig_powers, new_powers)
    memory.clear(warn=False)  # Clean up if the assertion passes

    return

