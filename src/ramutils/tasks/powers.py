from ramutils.log import get_logger
from ramutils.events import extract_subject
from ramutils.powers import reduce_powers as reduce_powers_core
from ramutils.powers import calculate_delta_hfa_table
from ramutils.powers import compute_normalized_powers as \
    compute_normalized_powers_core
from ramutils.powers import get_trigger_frequency_mask as \
    get_trigger_frequency_mask_core
from ramutils.controllability import calculate_modal_controllability, load_connectivity_matrix
from ramutils.tasks import task
from ramutils.powers import load_eeg as load_eeg_core
from functools import wraps

logger = get_logger()

__all__ = [
    'reduce_powers',
    'subset_powers',
    'create_target_selection_table',
    'compute_normalized_powers',
    'get_trigger_frequency_mask',
    'load_post_stim_eeg'
]


@task()
@wraps(reduce_powers_core)
def reduce_powers(powers, channel_mask, n_frequencies, frequency_mask=None):
    reduced_powers = reduce_powers_core(powers, channel_mask, n_frequencies,
                                        frequency_mask=frequency_mask)
    return reduced_powers


@task(cache=False)
def subset_powers(powers, mask):
    condensed_powers = powers[mask, :]
    return condensed_powers


@task(nout=2,cache=False)
@wraps(compute_normalized_powers_core)
def compute_normalized_powers(events, **kwargs):
    normalized_powers, updated_events = compute_normalized_powers_core(events,
                                                                       **kwargs)
    return normalized_powers, updated_events


@task()
def create_target_selection_table(pairs_metadata_table, normalized_powers,
                                  events, frequencies, hfa_cutoff=65,
                                  trigger_freq=110, root="/"):
    delta_hfa_table = calculate_delta_hfa_table(pairs_metadata_table,
                                                normalized_powers,
                                                events,
                                                frequencies,
                                                hfa_cutoff=hfa_cutoff,
                                                trigger_freq=trigger_freq)

    subject = extract_subject(events)
    connectivity_matrix = load_connectivity_matrix(subject, rhino_root=root)
    if connectivity_matrix is None:
        logger.warning("No DTI-based connectivity matrix found for %s" %
                       subject)
        modal_controllability_values = None

    else:
        coords = delta_hfa_table[['mni_x', 'mni_y', 'mni_z']].values
        modal_controllability_values = calculate_modal_controllability(
            connectivity_matrix, coords)

    delta_hfa_table['controllability'] = modal_controllability_values
    return delta_hfa_table


@task()
@wraps(get_trigger_frequency_mask_core)
def get_trigger_frequency_mask(trigger_frequency, frequencies):
    return get_trigger_frequency_mask_core(trigger_frequency, frequencies)


@task()
@wraps(load_eeg_core)
def load_post_stim_eeg(events, **kwargs):
    return load_eeg_core(events,
                         start_time=kwargs['post_stim_start_time'],
                         end_time=kwargs['post_stim_end_time'],
                         bipolar_pairs=kwargs.get('bipolar_pairs')
                         )
