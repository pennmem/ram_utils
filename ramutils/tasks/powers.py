import numpy as np

from ramutils.eeg.powers import compute_powers as compute_powers_core
from ramutils.classifier.utils import normalize_powers_by_session as \
    normalize_powers_by_session_core
from ramutils.tasks.events import get_encoding_mask, get_retrieval_mask
from ramutils.log import get_logger
from ramutils.tasks import task

logger = get_logger()


@task(nout=2)
def compute_powers(events, params):
    """

    :param np.recarray events:
    :param ExperimentParams params:
    :return: powers, updated_events

    """
    powers, updated_events = compute_powers_core(events,
                                                 params.start_time,
                                                 params.end_time,
                                                 params.buf,
                                                 params.freqs,
                                                 params.log_powers)
    return powers, updated_events


@task()
def combine_encoding_retrieval_powers(events, encoding_powers,
                                      retrieval_powers):
    encoding_mask = get_encoding_mask(events)
    retrieval_mask = get_retrieval_mask(events)

    powers = np.zeros((len(events), encoding_powers.shape[-1]))
    powers[encoding_mask, ...] = encoding_powers
    powers[retrieval_mask, ...] = retrieval_powers
    return powers

@task()
def normalize_powers_by_session(pow_mat, events):
    normalized_pow_mat = normalize_powers_by_session_core(pow_mat, events)
    return  normalized_pow_mat
