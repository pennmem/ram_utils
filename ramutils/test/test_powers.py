import pytest
import functools
import numpy as np
import pandas as pd

from pkg_resources import resource_filename

from ramutils.powers import reduce_powers, compute_single_session_powers, \
    compute_powers, reshape_powers_to_2d, reshape_powers_to_3d, \
    calculate_delta_hfa_table, compute_normalized_powers
from ramutils.tasks import memory
from ramutils.parameters import FRParameters, PALParameters
from ramutils.utils import load_event_test_data

datafile = functools.partial(resource_filename,
                             'ramutils.test.test_data.input')


@pytest.mark.rhino
def test_compute_single_session_powers(rhino_root):
    # Cases: EEG from monopolar, mixed mode, and bipolar. EEG with removed bad
    # data and without. Log powers true and false
    events = load_event_test_data(datafile(
        '/events/R1348J_task_events_rhino.npy'), rhino_root)[:5]
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
def test_compute_powers(event_file, rhino_root):
    # Cases: Bipolar_pairs == none and not none. Single session/multi.
    events_per_session = 5

    events = load_event_test_data(datafile('/events/' + event_file),
                                  rhino_root)

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
    reduced_power_matrices = [
        reduce_powers(sample_pow_mat, some_electrodes_masks[i], n_frequencies)
        for i in range(
        len(some_electrodes_masks))]
    assert all([reduced_power_matrices[i].shape == (n_events, expected_sizes[i]
                                                 * n_frequencies) for i in
                range(len(expected_sizes))])

    # Case 4: Select a subset of frequencies
    all_electrodes_mask = np.ones(100, dtype=bool)
    reduced_pow_mat = reduce_powers(sample_pow_mat, all_electrodes_mask,
                                    n_frequencies, )
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
    ('R1354E_task_events_rhino.npy', 'R1354E_normalized_powers.npy',
     'R1354E_hfa_ttest_table.csv', FRParameters)
])
def test_calculate_delta_hfa_table_regression(events, powers, exp_table,
                                              parameters, rhino_root):
    parameters = parameters().to_dict()
    powers = np.load(datafile('/powers/' + powers))
    events = load_event_test_data(datafile('/events/' + events), rhino_root)
    config_pairs = pd.read_csv(datafile('/montage/R1354E_montage_metadata.csv'),
                               index_col=0)
    hfa_table = calculate_delta_hfa_table(config_pairs, powers, events,
                                          parameters['freqs'],
                                          hfa_cutoff=65,
                                          trigger_freq=parameters['trigger_freq'])
    old_hfa_table = pd.read_csv(datafile('/powers/' + exp_table))

    assert np.allclose(old_hfa_table['t_stat'].values,
                       hfa_table['hfa_t_stat'].values)
    assert np.allclose(old_hfa_table['p_value'].values,
                       hfa_table['hfa_p_value'].values)

    return


@pytest.mark.rhino
@pytest.mark.slow
@pytest.mark.parametrize("events, exp_powers, parameters", [
    ('R1353N_task_events_rhino.npy', 'R1353N_normalized_powers.npy', PALParameters),
    ('R1354E_task_events_rhino.npy', 'R1354E_normalized_powers.npy', FRParameters),
    ('R1350D_task_events_rhino.npy', 'R1350D_normalized_powers.npy', FRParameters),
])
def test_regression_compute_normalized_powers(events, exp_powers, parameters,
                                              rhino_root):
    parameters = parameters().to_dict()
    events = load_event_test_data(datafile('/events/' + events), rhino_root)
    new_powers, updated_events = compute_normalized_powers(events,
                                                           **parameters)

    orig_powers = np.load(datafile('/powers/' + exp_powers))
    assert np.allclose(orig_powers, new_powers, atol=1e-1)
    memory.clear(warn=False)  # Clean up if the assertion passes

    return

