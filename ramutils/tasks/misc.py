from __future__ import unicode_literals

import os
import pandas as pd

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

from ptsa.data.readers import JsonIndexReader
from ramutils.io import store_results
from ramutils.utils import get_session_str

from ._wrapper import task
from ramutils.reports.summary import *
from ramutils.hmm import save_foresplot, save_traceplot
from ramutils.utils import is_stim_experiment as is_stim_experiment_core
from ramutils.utils import get_completed_sessions
from ramutils.utils import encode_file
from ramutils.utils import extract_experiment_series
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
                    behavioral_results=None, agg_report=False):
    """ Save all required output necessary to re-generate a report

    Parameters:
    -----------
    subject: str
        Subject ID
    experiment: str
        Experiment name
    session_summaries: List
        List of SessionSummary derived objects
    math_summaries: List
        List of MathSummary objects
    classifier_evaluation_results: List
        List of ClassifierSummary objects
    save_location: str
        Destination for data to be saved. Typically in
        /data10/RAM/report_database/ on RHINO
    retrained_classifier: ClassifierContainer
        Serialized representation of the retrained classifier
    target_selection_table pd.DataFrame
        DataFrame representation of the target selection table, formerly known
        as the subsequent memory effect table
    behavioral_results: dict
        Keys are the behavioral effect model type (stim list, stim item, etc.)
        and values are the traces from estimating those models

    Returns
    -------
    results_files: dict
        Dictionary whose keys are the names of statically-produced plots and
        values are encoded versions of those images. These are used to embed
        the static plots in the html reports during report generation

    Notes
    -----
    All output files are of the format {subject}_{experiment}_{session}_{data_type}.{file_type}
    where data_type is a generic name for the type of data being saved. The following data types
    map to a summary object:

    * sessions_summary: :class:`ramutils.reports.summary.SessionSummary`
    * math_summary: :class:`ramutils.reports.summary.MathSummary`
    * classifier_[tag]: :class:`ramutils.reports.summary.ClassifierSummary`

    """

    result_files = {}

    base_output_format = os.path.join(save_location,
                                      "{subject}_{experiment}_{session}_{"
                                      "data_type}.{file_type}")
    subject_specific_output = os.path.join(save_location,
                                      "{subject}_{data_type}.{file_type}")

    session_str = '_'.join([str(summary.session_number) for summary in
                            session_summaries])
    # Agg reports could have hundreds of sessions, so do not save them
    # as part of the file name
    if agg_report:
        session_str = ""

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
        # The retrained classifier is the same regardless of session/experiment
        # so just save one per subject
        retrained_classifier.save(subject_specific_output.format(
            subject=subject, data_type="retrained_classifier",
            file_type="zip"), overwrite=True)

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
                encoded_image = encode_file(f)
            result_files[name] = encoded_image

    return result_files


@task(cache=False)
def load_existing_results(subject, experiment, sessions, stim_report, db_loc,
                          joint_report, rootdir='/'):
    """ Load previously-saved data creating during report generation

    Parameters:
    -----------
    subject: str
        Subject ID
    experiment: str
        Experiment ID
    sessions: list or None
        If none, then sessions are looked up from r1.json for the given subject and experiment.
    stim_report: bool
        Indicator for if the requested data is associated with a stim report
    db_loc: str
        Report database location relative to rootdir. db_loc will be appended to rootdir
        to find the full absolute path. If both db_loc and rootdir are absolute paths,
        it will be assumed that db_loc contains the root directory.
    rootdir: str
        RHINO mount point or root directory

    Returns:
    --------
    saved_results: dict
        Mirrors the input to save_all_output

    """

    saved_results = {
        'target_selection_table': None,
        'classifier_evaluation_results': None,
        'session_summaries': None,
        'math_summaries': None,
        'hmm_results': None
    }

    # Repetition ratio dictionary optional
    # Cases: PS, stim, non-stim
    subject_experiment = "_".join([subject, experiment])
    base_output_format = os.path.join(rootdir, db_loc, subject_experiment +
                                      "_{session}_{data_type}.{file_type}")

    if sessions is None:
        if joint_report and 'FR' in experiment:
            series_num = extract_experiment_series(experiment)
            fr_sessions = get_completed_sessions(subject, 'FR'+series_num,
                                                 rootdir)
            catfr_sessions = get_completed_sessions(subject, 'catFR'+series_num,
                                                    rootdir)
            catfr_sessions = set(str(100 + int(s)) for s in catfr_sessions)
            sessions = fr_sessions | catfr_sessions
        else:
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
            saved_results['target_selection_table'] = target_selection_table

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
            saved_results['classifier_evaluation_results'] = classifier_evaluation_results

            session_summaries, math_summaries = [], []
            for session in sessions:
                math_summary = MathSummary.from_hdf(
                    base_output_format.format(session=str(session),
                                              data_type='math_summary',
                                              file_type='h5')
                )
                math_summaries.append(math_summary)
                if (experiment == 'catFR1') or (int(session) >= 100):
                    summary = CatFRSessionSummary
                elif experiment == 'FR1':
                    summary = FRSessionSummary

                session_summary = summary.from_hdf(
                    base_output_format.format(session=str(session),
                                              data_type='session_summary',
                                              file_type='h5'))
                session_summaries.append(session_summary)

            saved_results['session_summaries'] = session_summaries
            saved_results['math_summaries'] = math_summaries

        elif stim_report and 'PS' not in experiment:
            classifier_evaluation_results, math_summaries, session_summaries = [], [], []
            for session in sessions:
                classifier_summary = ClassifierSummary.from_hdf(
                    base_output_format.format(session=str(session),
                                              data_type='classifier_session_' +
                                              str(session),
                                              file_type='h5'))
                classifier_evaluation_results.append(classifier_summary)

                math_summary = MathSummary.from_hdf(
                    base_output_format.format(session=str(session),
                                              data_type='math_summary',
                                              file_type='h5'))
                math_summaries.append(math_summary)

                if 'TICL' in experiment:
                    klass = TICLFRSessionSummary
                else:
                    klass = FRStimSessionSummary
                session_summary = klass.from_hdf(
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
                            data_type=(name + '_foresplot'),
                            file_type='png')
                        assert os.path.exists(forestplot_path)

                        # Encode the image and pass along that data
                        with open(forestplot_path, 'rb') as f:
                            encoded_image = encode_file(f)
                        hmm_results[name] = encoded_image
                    saved_results['hmm_results'] = hmm_results

            saved_results['session_summaries'] = session_summaries
            saved_results['math_summaries'] = math_summaries
            saved_results['classifier_evaluation_results'] = classifier_evaluation_results

        else:
            return saved_results

    except (IOError, OSError, AssertionError):
        logger.warning('Not all underlying data could be found for the '
                       'requested report, building from scratch instead.')
        return saved_results

    return saved_results
