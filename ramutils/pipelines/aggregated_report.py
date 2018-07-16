""" Pipeline for creating aggregated stim reports """

from ramutils.tasks import *
from ramutils.events import find_subjects
from ramutils.log import get_logger
from .hooks import PipelineCallback

logger = get_logger()


def make_aggregated_report(subjects=None, experiments=None, sessions=None,
                           fit_model=True, paths=None, pipeline_name="aggregate"):
    """ Build an aggregated stim session report

    This pipeline should be used for combining data across stim experiment sessions into a single report. The concept
    of a "joint report" already exists for record-only sessions and can be generated using the
    `ramutils.pipelines.report.make_report` pipeline. In the future, a more sensible approach would be to have joint
    reports for both stim sessions and record-only sessions be built using the same pipeline.


    Keyword Arguments
    -----------------
    subjects: list or None
        The set of subjects to include when building the report. If None and one or more experiments are specified, then
        the subjects who completed each experiment will be identified automatically.

    experiments: list or None
        The set of experiments to include when building the report. This is primarily useful for combining FR with CatFR
        session data, effectively a joint report for stim sessions. However, it is also possible to combine across
        experiment series. For example, a joint report could be built that combines FR3, catFR3, FR5, and catFR5. This
        is possible because the report templates for these experiments are identical. It is not, however, possible to
        combine stim reports across dissimilar reports. For example, it would not make sense to build an aggregate report
        combining PS5 with catFR5 since those use completely different templates.

    sessions: list or None
        The set of sessions to include. This parameter can only be used if a single subject and a single experiment have
        been provided. The main use case is for generating a stim report that excludes 1 or more sessions. We do not
        currently support the ability to combine the sessions paramter with more than one subject or more than one
        experiment. This could be a future enhancement. For example, it may be useful to be able to generate an
        aggregated report of all the first sessions of a particular experiment type, or all first sessions for a
        particular subject.

    fit_model: bool
        If true, the a Bayesian hierachical multilvel model will be fit using the data combined across the requested
        subjects, experiments, and sessions. This process can be very slow as the number of sessions increases, so it
        is False by default. The main use case if for building a stim report that aggregates over the sessions that
        a particular subject completed of a particular experiment.

    paths: `ramutils.parameters.FilePaths`
        Helper class for setting up the set of paths that will be necessary for loading existing results

    pipeline_name : str
        Name to use for status updates.

    """
    if experiments is not None:
        for i, experiment in enumerate(experiments):
            if 'Cat' in experiment:
                experiments[i] = experiments[i].replace('Cat', 'cat')

    all_classifier_evaluation_results, all_session_summaries, all_math_summaries, target_selection_table = [], [], [], None

    # All subjects completing a given experiment(s)
    if subjects is None and experiments is not None:
        for experiment in experiments:
            exp_subjects = find_subjects(experiment, paths.root)
            for subject in exp_subjects:
                pre_built_results = load_existing_results(subject, experiment, sessions, True,
                                                          paths.data_db,
                                                          joint_report=False,
                                                          rootdir=paths.root).compute()
                if all([val is None for val in [pre_built_results['classifier_evaluation_results'],
                                                pre_built_results['session_summaries'],
                                                pre_built_results['math_summaries']]]):
                    logger.warning('Unable to find underlying data for {}, experiment {}'.format(
                        subject, experiment))
                    continue

                all_classifier_evaluation_results.extend(
                    pre_built_results['classifier_evaluation_results'])
                all_session_summaries.extend(
                    pre_built_results['session_summaries'])
                all_math_summaries.extend(pre_built_results['math_summaries'])
        subject = 'combined'
        experiment = "_".join(experiments)

    # Set of subject(s) completing a specific set of experiment(s)
    elif subjects is not None and experiments is not None and sessions is None:
        for subject in subjects:
            for experiment in experiments:
                pre_built_results = load_existing_results(subject, experiment, sessions, True,
                                                          paths.data_db,
                                                          joint_report=False,
                                                          rootdir=paths.root).compute()
                # Check if only None values were returned. Processing will continue
                # undeterred
                if all([val is None for val in [pre_built_results['classifier_evaluation_results'],
                                                pre_built_results['session_summaries'],
                                                pre_built_results['math_summaries']]]):
                    logger.warning('Unable to find underlying data for {}, experiment {}'.format(
                        subject, experiment))
                    continue

                all_classifier_evaluation_results.extend(
                    pre_built_results['classifier_evaluation_results'])
                all_session_summaries.extend(
                    pre_built_results['session_summaries'])
                all_math_summaries.extend(pre_built_results['math_summaries'])
        subject = '_'.join(subjects)
        experiment = "_".join(experiments)

    # Single subject/experiment and a subset of sessions
    elif subjects is not None and experiments is not None and sessions is not None:
        if len(subjects) > 1 or len(experiments) > 1:
            raise RuntimeError(
                "When specifying sessions, only single subject and experiment are allowed")
        subject = subjects[0]
        experiment = experiments[0]

        pre_built_results = load_existing_results(subject, experiment, sessions, True,
                                                  paths.data_db,
                                                  joint_report=False,
                                                  rootdir=paths.root).compute()
        if all([val is None for val in [pre_built_results['classifier_evaluation_results'],
                                        pre_built_results['session_summaries'],
                                        pre_built_results['math_summaries']]]):
            logger.warning('Unable to find underlying data for {}, experiment {}'.format(
                subject, experiment))

        all_classifier_evaluation_results.extend(
            pre_built_results['classifier_evaluation_results'])
        all_session_summaries.extend(pre_built_results['session_summaries'])
        all_math_summaries.extend(pre_built_results['math_summaries'])

    else:
        raise RuntimeError(
            'The requested type of aggregation is not currently supported')

    if fit_model:
        # Fit model and save resulting images. subject/experiment do not really matter since the model is the same
        # at least for now
        hmm_results = estimate_effects_of_stim(
            subject, 'FR3', all_session_summaries)
        output = save_all_output(subject, experiment, [], [], [], paths.data_db,
                                 behavioral_results=hmm_results, agg_report=True)
    else:
        output = None

    report = build_static_report(subject, experiment, all_session_summaries,
                                 all_math_summaries, target_selection_table,
                                 all_classifier_evaluation_results,
                                 paths.dest, hmm_results=output,
                                 save=True, aggregated_report=True)

    with PipelineCallback(pipeline_name):
        return report.compute()
