import numpy as np

from ramutils.eeg.powers import compute_powers as compute_powers_core
from ramutils.log import get_logger
from ramutils.tasks import task

logger = get_logger()


@task()
def compute_powers(events, params):
    """

    :param np.recarray events:
    :param ExperimentParams params:
    :return: powers

    """
    powers, events_ = compute_powers_core(events, params.start_time, params.end_time,
                                          params.buf, params.freqs, params.log_powers)
    return powers
