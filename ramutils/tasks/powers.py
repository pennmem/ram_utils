import numpy as np

from ramutils.eeg.powers import compute_normalized_powers as \
    compute_normalized_powers_core
from ramutils.eeg.powers import reduce_powers as reduce_powers_core
from ramutils.log import get_logger
from ramutils.tasks import task

logger = get_logger()

__all__ = [
    'compute_normalized_powers',
    'reduce_powers'
]


@task(nout=2)
def compute_normalized_powers(events, start_time, end_time, buffer_time,
                              frequencies, log_powers, filt_order, width):
    powers, updated_events = compute_normalized_powers_core(events,
                                                            start_time,
                                                            end_time,
                                                            buffer_time,
                                                            frequencies,
                                                            log_powers,
                                                            filt_order=filt_order,
                                                            width=width)
    return powers, updated_events


@task()
def reduce_powers(powers, mask, n_frequencies):
    reduced_powers = reduce_powers_core(powers, mask, n_frequencies)
    return reduced_powers
