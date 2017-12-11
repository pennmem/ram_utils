import os
from ptsa.data.readers import JsonIndexReader
from ._wrapper import task
from ramutils.utils import is_stim_experiment as is_stim_experiment_core


__all__ = [
    'read_index',
    'is_stim_experiment'
]


@task()
def read_index(mount_point='/'):
    """Reads the JSON index reader.

    :param str mount_point: Root directory to search for.
    :returns: JsonIndexReader

    """
    path = os.path.join(mount_point, 'protocols', 'r1.json')
    return JsonIndexReader(path)


@task(cache=False)
def is_stim_experiment(experiment):
    is_stim = is_stim_experiment_core(experiment)
    return is_stim

