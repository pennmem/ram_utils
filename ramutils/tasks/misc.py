from __future__ import unicode_literals

import os
import base64
import pandas as pd

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

from ptsa.data.readers import JsonIndexReader
from ramutils.io import load_results, store_results
from ramutils.utils import get_session_str

from ._wrapper import task
from ramutils.reports.summary import *
from ramutils.hmm import save_foresplot, save_traceplot
from ramutils.utils import is_stim_experiment as is_stim_experiment_core
from ramutils.utils import get_completed_sessions
from ramutils.log import get_logger

logger = get_logger()


__all__ = [
    'read_index',
    'is_stim_experiment',
    'save_all_output',
    'load_existing_results'
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
                    classifier_evaluation_results, save_location,
                    retrained_classifier=None, target_selection_table=None,
                    behavioral_results=None):

    result_files = {}

    base_output_format = os.path.join(save_location,
                                      "{subject}_{experiment}_{session}_{"
                                      "data_type}.{file_type}")

    session_str = '_'.join([str(summary.session_number) for summary in
                            session_summaries])

    if (target_selection_table is not None) and \
            (len(target_selection_table) > 0):
        target_selection_table.to_csv(
            base_output_format.format(subject=subject,
                                      experiment=experiment,
                                      session=session_str,
                                      data_type='target_selection_table',
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

    for classifier_summary in classifier_evaluation_results:
        sessions = classifier_summary.sessions
        session_str = get_session_str(sessions)
        store_results(classifier_summary, base_output_format.format(
            subject=subject, experiment=experiment, session=session_str,
            data_type='classifier_' + classifier_summary.tag,
            file_type='h5'))

    if retrained_classifier is not None:
        retrained_classifier.save(base_output_format.format(
            subject=subject, experiment=experiment, session='all',
            data_type="retrained_classifier", file_type="zip"), overwrite=True)

    # Save plots from hmm models and return file paths in a dict
    if behavioral_results is not None:
        for name, trace in behavioral_results.items():
            forestplot_path = base_output_format.format(subject=subject,
                                                        experiment=experiment,
                                                        session=session_str,
                                                        data_type=(name +
                                                                   '_foresplot'),
                                                        file_type='png')
            save_foresplot(trace, forestplot_path)
            traceplot_path = base_output_format.format(subject=subject,
                                                       experiment=experiment,
                                                       session=session_str,
                                                       data_type=(name +
                                                                  '_traceplot'),
                                                       file_type='png')
            save_traceplot(trace, traceplot_path)

            with open(forestplot_path, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).replace(b"\n", b"").decode()
            result_files[name] = encoded_image

    return result_files


@task(cache=False)
def load_existing_results(subject, experiment, sessions, stim_report, db_loc,
                          rootdir='/'):
    # Repetition ratio dictionary optional
    # Cases: PS, stim, non-stim
    subject_experiment = "_".join([subject, experiment])
    base_output_format = os.path.join(db_loc, subject_experiment +
                                      "_{session}_{data_type}.{file_type}")

    if sessions is None:
        sessions = get_completed_sessions(subject, experiment, rootdir=rootdir)

    session_str = get_session_str(sessions)
    target_selection_table = None
    hmm_results = {}
    try:
        if stim_report is False:
            target_selection_table = pd.read_csv(
                base_output_format.format(session=session_str,
                                          data_type='target_selection_table',
                                          file_type='csv'))
            encoding_classifier_summary = ClassifierSummary.from_hdf(
                base_output_format.format(session=session_str,
                                          data_type='classifier_Encoding',
                                          file_type='h5'))
            joint_classifier_summary = ClassifierSummary.from_hdf(
                base_output_format.format(session=session_str,
                                          data_type='classifier_Joint',
                                          file_type='h5'))
            classifier_evaluation_results = [encoding_classifier_summary,
                                             joint_classifier_summary]
            session_summaries, math_summaries = [], []
            for session in sessions:
                math_summary = MathSummary.from_hdf(
                    base_output_format.format(session=str(session),
                                              data_type='math_summary',
                                              file_type='h5')
                )
                math_summaries.append(math_summary)
                if (experiment == 'catFR1') or (session >= 100):
                    summary = CatFRSessionSummary
                elif experiment == 'FR1':
                    summary = FRSessionSummary

                session_summary = summary.from_hdf(
                    base_output_format.format(session=str(session),
                                              data_type='session_summary',
                                              file_type='h5'))
                session_summaries.append(session_summary)

        elif stim_report and 'PS' not in experiment:
            classifier_evaluation_results, math_summaries, session_summaries = [], [], []
            for session in sessions:
                classifier_summary = ClassifierSummary.from_hdf(
                    base_output_format.format(session=str(session),
                                              data_type='classifier_session_' + str(session),
                                              file_type='h5'))
                classifier_evaluation_results.append(classifier_summary)

                math_summary = MathSummary.from_hdf(
                    base_output_format.format(session=str(session),
                                              data_type='math_summary',
                                              file_type='h5'))
                math_summaries.append(math_summary)

                session_summary = FRStimSessionSummary.from_hdf(
                    base_output_format.format(session=str(session),
                                              data_type='session_summary',
                                              file_type='h5'))
                session_summaries.append(session_summary)

                # Check if behavioral model results are saved
                if 'FR5' in experiment:
                    for name in ['list', 'stim_item', 'post_stim_item']:
                        forestplot_path = base_output_format.format(
                            subject=subject,
                            experiment=experiment,
                            session=str(session),
                            data_type=(name +'_foresplot'),
                            file_type='png')
                        assert os.path.exists(forestplot_path)

                        # Encode the image and pass along that data
                        with open(forestplot_path, 'rb') as f:
                            encoded_image = base64.b64encode(f.read()).replace(b"\n", b"").decode()
                        hmm_results[name] = encoded_image

        else:
            return None, None, None, None, None

    except (IOError, OSError, AssertionError):
        logger.warning('Not all underlying data could be found for the '
                       'requested report, building from scratch instead.')
        return None, None, None, None, None

    return target_selection_table, classifier_evaluation_results, \
           session_summaries, math_summaries, hmm_results

