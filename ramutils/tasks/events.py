"""Partial pipelines for processing events that is used by full pipelines."""

from ramutils.events import load_events, clean_events, select_word_events, \
    concatenate_events_across_experiments
from ramutils.tasks import task
from ramutils.utils import extract_experiment_series

__all__ = [
    'build_test_data',
    'build_training_data'
]


@task()
def build_training_data(subject, experiment, paths, sessions=None, **kwargs):
    """ Construct the set of events needed for classifier training """
    if "PAL" in experiment:
        pal_events = load_events(subject, "PAL1", sessions=sessions,
                                 rootdir=paths.root)
        cleaned_pal_events = clean_events(pal_events)

    if (("FR" in experiment) and kwargs['combine_events']) or \
            ("PAL" in experiment and kwargs['combined_events']):
        fr_events = load_events(subject, 'FR1', sessions=sessions,
                                rootdir=paths.root)
        cleaned_fr_events = clean_events(fr_events,
                                         start_time=kwargs['baseline_removal_start_time'],
                                         end_time=kwargs['retrieval_time'],
                                         duration=kwargs['empty_epoch_duration'],
                                         pre=kwargs['pre_event_buf'],
                                         post=kwargs['post_event_buf'])

        catfr_events = load_events(subject, 'catFR1',
                                   sessions=sessions, rootdir=paths.root)
        cleaned_catfr_events = clean_events(catfr_events,
                                            start_time=kwargs['baseline_removal_start_time'],
                                            end_time=kwargs['retrieval_time'],
                                            pre=kwargs['pre_event_buf'],
                                            post=kwargs['post_event_buf'],
                                            duration=kwargs['empty_epoch_duration'])

        free_recall_events = concatenate_events_across_experiments(
            [cleaned_fr_events, cleaned_catfr_events])

    elif "FR" in experiment and not kwargs['combine_events']:
        free_recall_events = load_events(subject, experiment, sessions=sessions,
                                         rootdir=paths.root)
        free_recall_events = clean_events(free_recall_events,
                                          start_time=kwargs['baseline_removal_start_time'],
                                          end_time=kwargs['retrieval_time'],
                                          duration=kwargs['empty_epoch_duration'],
                                          pre=kwargs['pre_event_buf'],
                                          post=kwargs['post_event_buf'])

    if ("PAL" in experiment) and kwargs['combine_events']:
        all_task_events = concatenate_events_across_experiments([
            free_recall_events, cleaned_pal_events])

    elif ("PAL" in experiment) and not kwargs['combine_events']:
        all_task_events = cleaned_pal_events

    else:
        all_task_events = free_recall_events

    all_task_events = select_word_events(all_task_events,
                                         encoding_only=kwargs['encoding_only'])
    return all_task_events


@task(nout=2)
def build_test_data(subject, experiment, paths, joint_report, sessions=None,
                    **kwargs):
    """
        Construct the set of events to be used for post-hoc classifier
        evaluation, i.e. the test data

    """
    series_num = extract_experiment_series(experiment)
    if joint_report and 'FR' in experiment:
        fr_events = load_events(subject, 'FR' + series_num,
                                sessions=sessions, rootdir=paths.root)
        cleaned_fr_events = clean_events(fr_events,
                                         start_time=kwargs['baseline_removal_start_time'],
                                         end_time=kwargs['retrieval_time'],
                                         duration=kwargs['empty_epoch_duration'],
                                         pre=kwargs['pre_event_buf'],
                                         post=kwargs['post_event_buf'])

        catfr_events = load_events(subject, 'catFR' + series_num,
                                   sessions=sessions,
                                   rootdir=paths.root)
        cleaned_catfr_events = clean_events(catfr_events,
                                            start_time=kwargs['baseline_removal_start_time'],
                                            end_time=kwargs['retrieval_time'],
                                            duration=kwargs['empty_epoch_duration'],
                                            pre=kwargs['pre_event_buf'],
                                            post=kwargs['post_event_buf'])

        all_events = concatenate_events_across_experiments([fr_events,
                                                            catfr_events])
        task_events = concatenate_events_across_experiments(
            [cleaned_fr_events, cleaned_catfr_events], stim=True)

    elif not joint_report and 'FR' in experiment:
        all_events = load_events(subject, experiment, sessions=sessions,
                                 rootdir=paths.root)
        task_events = clean_events(all_events,
                                   start_time=kwargs['baseline_removal_start_time'],
                                   end_time=kwargs['retrieval_time'],
                                   duration=kwargs['empty_epoch_duration'],
                                   pre=kwargs['pre_event_buf'],
                                   post=kwargs['post_event_buf'])

    else:
        all_events = load_events(subject, experiment, sessions=sessions,
                                 rootdir=paths.root)
        task_events = clean_events(all_events)

    return all_events, task_events

