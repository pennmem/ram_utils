import numpy as np

from ramutils.events import partition_events, \
    concatenate_events_for_single_experiment
from ramutils.log import get_logger
from ramutils.powers import compute_powers
from ramutils.powers import reduce_powers as reduce_powers_core
from ramutils.tasks import task

logger = get_logger()

__all__ = [
    'compute_normalized_powers',
    'reduce_powers'
]


@task()
def reduce_powers(powers, mask, n_frequencies):
    reduced_powers = reduce_powers_core(powers, mask, n_frequencies)
    return reduced_powers


@task(nout=2)
def compute_normalized_powers(events, **kwargs):
    """ Compute powers by session, encoding/retrieval, and FR vs. PAL

    Notes
    -----
    There are different start times, end time, and buffer times for each
    subset type, so those are passed in as kwargs and looked up prior to
    calling the more general compute_powers function

    """

    event_partitions = partition_events(events)
    power_partitions = []
    cleaned_event_partitions = []
    for subset_name, event_subset in event_partitions.items():
        if len(event_subset) == 0:
            continue

        if subset_name == 'fr_encoding':
            start_time = kwargs['encoding_start_time']
            end_time = kwargs['encoding_end_time']
            buffer_time = kwargs['encoding_buf']

        elif subset_name == 'fr_retrieval':
            start_time = kwargs['retrieval_start_time']
            end_time = kwargs['retrieval_end_time']
            buffer_time = kwargs['retrieval_buf']

        elif subset_name == 'pal_encoding':
            start_time = kwargs['pal_start_time']
            end_time = kwargs['pal_end_time']
            buffer_time = kwargs['pal_buf_time']

        elif subset_name == 'pal_retrieval':
            start_time = kwargs['pal_retrieval_start_time']
            end_time = kwargs['pal_retrieval_end_time']
            buffer_time = kwargs['pal_retrieval_buf']

        else:
            raise RuntimeError("Unexpected event subset was encountered")

        powers, cleaned_events = compute_powers(event_subset,
                                                start_time,
                                                end_time,
                                                buffer_time,
                                                kwargs['freqs'],
                                                kwargs['log_powers'],
                                                filt_order=kwargs['filt_order'],
                                                width=kwargs['width'])
        cleaned_event_partitions.append(cleaned_events)
        power_partitions.append(powers)

    cleaned_events = concatenate_events_for_single_experiment(
        cleaned_event_partitions)
    combined_powers = np.concatenate(power_partitions)

    return combined_powers, cleaned_events