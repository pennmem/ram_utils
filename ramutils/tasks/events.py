"""Partial pipelines for processing events that is used by full pipelines."""

from ramutils.events import load_events, clean_events, select_word_events, \
    concatenate_events_across_experiments
from ramutils.events import get_word_event_mask as get_word_event_mask_core
from ramutils.events import get_repetition_ratio_dict as \
    get_repetition_ratio_dict_core
from ramutils.events import get_post_stim_events_mask as \
    get_post_stim_events_mask_core
from ramutils.events import remove_practice_lists, remove_sessions
from ramutils.tasks import task
from ramutils.utils import extract_experiment_series

__all__ = [
    'get_word_event_mask',
    'subset_events',
    'build_test_data',
    'build_training_data',
    'get_repetition_ratio_dict',
    'get_post_stim_events_mask',
    'build_ps_data'
]


@task()
def get_word_event_mask(events, encoding_only):
    return get_word_event_mask_core(events, encoding_only)


@task()
def get_post_stim_events_mask(events):
    return get_post_stim_events_mask_core(events)


@task()
def subset_events(events, mask):
    events_subset = events[mask]
    return events_subset


@task()
def build_training_data(subject, experiment, paths, sessions=None, excluded_sessions=None, **kwargs):
    """ Construct the set of events needed for classifier training """
    if "PAL" in experiment:
        pal_events = load_events(subject, "PAL1", sessions=sessions,
                                 rootdir=paths.root)
        cleaned_pal_events = clean_events(pal_events)
        cleaned_pal_events = remove_sessions(cleaned_pal_events, excluded_sessions)

    if (("FR" in experiment) and kwargs['combine_events']) or \
            ("PAL" in experiment and kwargs['combine_events']):
        fr_events = load_events(subject, 'FR1', sessions=sessions,
                                rootdir=paths.root)
        cleaned_fr_events = clean_events(fr_events,
                                         start_time=kwargs['baseline_removal_start_time'],
                                         end_time=kwargs['retrieval_time'],
                                         duration=kwargs['empty_epoch_duration'],
                                         pre=kwargs['pre_event_buf'],
                                         post=kwargs['post_event_buf'])

        catfr_events = load_events(subject, 'catFR1',
                                   sessions=sessions,
                                   rootdir=paths.root)
        cleaned_catfr_events = clean_events(catfr_events,
                                            start_time=kwargs['baseline_removal_start_time'],
                                            end_time=kwargs['retrieval_time'],
                                            duration=kwargs['empty_epoch_duration'],
                                            pre=kwargs['pre_event_buf'],
                                            post=kwargs['post_event_buf'])

        free_recall_events = concatenate_events_across_experiments(
            [cleaned_fr_events, cleaned_catfr_events], cat=True)
        free_recall_events = remove_sessions(free_recall_events, excluded_sessions)

    elif "FR" in experiment and not kwargs['combine_events']:
        free_recall_events = load_events(subject, experiment, sessions=sessions,
                                         rootdir=paths.root)
        free_recall_events = clean_events(free_recall_events,
                                          start_time=kwargs['baseline_removal_start_time'],
                                          end_time=kwargs['retrieval_time'],
                                          duration=kwargs['empty_epoch_duration'],
                                          pre=kwargs['pre_event_buf'],
                                          post=kwargs['post_event_buf'])
        free_recall_events = remove_sessions(free_recall_events, excluded_sessions)

    if ("PAL" in experiment) and kwargs['combine_events']:
        all_task_events = concatenate_events_across_experiments([
            free_recall_events, cleaned_pal_events])
        all_task_events = remove_sessions(all_task_events, excluded_sessions)

    elif ("PAL" in experiment) and not kwargs['combine_events']:
        all_task_events = cleaned_pal_events

    else:
        all_task_events = free_recall_events

    all_task_events = select_word_events(all_task_events,
                                         encoding_only=kwargs['encoding_only'])

    if len(all_task_events) == 0:
        raise RuntimeError("No events found")

    return all_task_events


@task(nout=3)
def build_test_data(subject, experiment, paths, joint_report, sessions=None,
                    **kwargs):
    """
        Construct the set of events to be used for post-hoc classifier
        evaluation, i.e. the test data
    """
    series_num = extract_experiment_series(experiment)
    if joint_report and 'FR' in experiment:
        fr_events = load_events(subject, 'FR' + series_num,
                                sessions=sessions,
                                rootdir=paths.root)
        cleaned_fr_events, fr_stim_params = clean_events(
            fr_events, start_time=kwargs['baseline_removal_start_time'],
            end_time=kwargs['retrieval_time'],
            duration=kwargs['empty_epoch_duration'],
            pre=kwargs['pre_event_buf'], post=kwargs['post_event_buf'],
            return_stim_events=True)

        catfr_events = load_events(subject, 'catFR' + series_num,
                                   sessions=sessions,
                                   rootdir=paths.root)
        cleaned_catfr_events, catfr_stim_params = clean_events(
            catfr_events, start_time=kwargs['baseline_removal_start_time'],
            end_time=kwargs['retrieval_time'],
            duration=kwargs['empty_epoch_duration'],
            pre=kwargs['pre_event_buf'], post=kwargs['post_event_buf'],
            return_stim_events=True)

        all_events = concatenate_events_across_experiments([fr_events,
                                                            catfr_events])
        task_events = concatenate_events_across_experiments(
            [cleaned_fr_events, cleaned_catfr_events], cat=True)

        stim_params = concatenate_events_across_experiments([fr_stim_params,
                                                             catfr_stim_params],
                                                            stim=True)

    elif not joint_report and 'FR' in experiment:
        all_events = load_events(subject, experiment, sessions=sessions,
                                 rootdir=paths.root)
        task_events, stim_params = clean_events(
            all_events, start_time=kwargs['baseline_removal_start_time'],
            end_time=kwargs['retrieval_time'],
            duration=kwargs['empty_epoch_duration'],
            pre=kwargs['pre_event_buf'], post=kwargs['post_event_buf'],
            return_stim_events=True)

    else:
        all_events = load_events(subject, experiment, sessions=sessions,
                                 rootdir=paths.root)
        task_events, stim_params = clean_events(all_events,
                                                return_stim_events=True)

    # Clean all events after using them to build task events because cleaning
    #  will remove fields that have nested recarrays to make serialization of
    #  these events possible downstream
    if series_num != '1':
        all_events = clean_events(all_events, all_events=True)

    if len(all_events) == 0:
        raise RuntimeError('No events found')

    return all_events, task_events, stim_params


@task()
def get_repetition_ratio_dict(paths):
    return get_repetition_ratio_dict_core(rootdir=paths.root)


@task()
def build_ps_data(subject, experiment, file_type, sessions, rootdir):
    updated_experiment = experiment.replace("PS4_", "")
    ps_events = load_events(subject, updated_experiment, file_type=file_type,
                            sessions=sessions, rootdir=rootdir)
    # The practice list is needed in order assess sham stim event, so this
    # may need to change in the future
    ps_events = remove_practice_lists(ps_events)

    if len(ps_events) == 0:
        raise RuntimeError("No events found")
    return ps_events

