import os
import pandas as pd

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

from ptsa.data.readers import JsonIndexReader
from ramutils.io import load_results, store_results

from ._wrapper import task
from ramutils.utils import is_stim_experiment as is_stim_experiment_core


__all__ = [
    'read_index',
    'is_stim_experiment',
    'save_all_output'
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


@task(cache=False)
def save_all_output(subject, experiment, session_summaries, math_summaries,
                    delta_hfa_table, classifier_evaluation_results,
                    save_location):

    base_output_format = os.path.join(save_location,
                                      "{subject}_{experiment}_{session}_{"
                                      "data_type}.{file_type}")

    delta_hfa_table.to_csv(base_output_format.format(subject=subject,
                                                     experiment=experiment,
                                                     session='all',
                                                     data_type='delta_hfa_table',
                                                     file_type='csv'))

    for session_summary in session_summaries:
        session = session_summary.session_number
        store_results(session_summary, base_output_format.format(
            subject=subject, experiment=experiment, session=session,
            data_type='session_summary', file_type='h5'))

    for math_summary in math_summaries:
        session = math_summary.session_number
        store_results(math_summary, base_output_format.format(
            subject=subject, experiment=experiment, session=session,
            data_type='math_summary', file_type='h5'))

    for classifier_results in classifier_evaluation_results:
        for classifier_summary in classifier_results:
            session = classifier_summary.sessions
            store_results(classifier_summary, base_output_format.format(
                subject=subject, experiment=experiment, session=session,
                data_type='classifier_summary', file_type='h5'))

    return True

