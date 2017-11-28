import numpy as np

from ramutils.powers import reduce_powers




# Cases: EEG from monopolar, mixed mode, and bipolar. EEG with removed bad
# data and without. Log powers true and false
def test_compute_single_session_powers():
    return

# Cases: Bipolar_pairs == none and not none
def test_compute_powers():
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


# Not sure how much there is to test here
def test_normalize_powers_by_session():
    return


def test_regression_compute_normalized_powers():
    return

