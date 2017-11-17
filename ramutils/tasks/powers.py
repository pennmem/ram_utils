import numpy as np

from ramutils.eeg.powers import compute_powers as compute_powers_core
from ramutils.classifier.utils import normalize_powers_by_session as \
    normalize_powers_by_session_core
from ramutils.tasks.events import get_encoding_mask, get_all_retrieval_events_mask
from ramutils.log import get_logger
from ramutils.tasks import task

logger = get_logger()

__all__ = [
    'compute_powers',
    'combine_encoding_retrieval_powers',
    'normalize_powers_by_session',
    'reduce_powers',
]


@task(nout=2)
def compute_powers(events, start_time, end_time, buffer_time, frequencies,
                   log_powers, filt_order, width):
    """ Compute powers. See documentation for compute_powers_core for more
    info on kwargs and what is being done
    """
    powers, updated_events = compute_powers_core(events,
                                                 start_time,
                                                 end_time,
                                                 buffer_time,
                                                 frequencies,
                                                 log_powers,
                                                 filt_order=filt_order,
                                                 width=width)
    return powers, updated_events


@task()
def combine_encoding_retrieval_powers(events, encoding_powers,
                                      retrieval_powers):
    encoding_mask = get_encoding_mask(events)
    retrieval_mask = get_all_retrieval_events_mask(events)

    powers = np.zeros((len(events), encoding_powers.shape[-1]))
    powers[encoding_mask, ...] = encoding_powers
    powers[retrieval_mask, ...] = retrieval_powers
    return powers


@task()
def normalize_powers_by_session(pow_mat, events):
    normalized_pow_mat = normalize_powers_by_session_core(pow_mat, events)
    return  normalized_pow_mat


@task()
def reduce_powers(powers, mask, n_frequencies):
    """ Create a subset of the full power matrix by excluding certain electrodes

    Parameters
    ----------
    powers: np.ndarray
        Original power matrix
    mask: array_like
        Boolean array of size n_channels
    n_frequencies: int
        Number of frequencies used in calculating the power matrix. This is
        needed to be able to properly reshape the array

    Returns
    -------
    np.ndarray
        Subsetted power matrix

    """
    # Reshape into 3-dimensional array (n_events, n_electrodes, n_frequencies)
    reduced_powers = powers.reshape((len(powers), -1, n_frequencies))
    reduced_powers = reduced_powers[:, mask, :]

    # Reshape back to 2D representation so it can be used as a feature matrix
    reduced_powers = reduced_powers.reshape((len(reduced_powers), -1))

    return reduced_powers
