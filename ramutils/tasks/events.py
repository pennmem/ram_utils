"""Partial pipelines for processing events that is used by full pipelines."""

from ramutils.events import select_word_events as select_word_events_core
from ramutils.events import concatenate_events_across_experiments as concatenate_events_across_experiments_core
from ramutils.events import load_events as load_events_core
from ramutils.events import clean_events as clean_events_core
from ramutils.tasks import task

__all__ = [
    'load_events',
    'clean_events',
    'concatenate_events_across_experiments',
    'select_word_events'
]


@task()
def load_events(subject, experiment, sessions=None, rootdir='/'):
    events = load_events_core(subject, experiment, sessions=sessions, rootdir=rootdir)
    return events


@task()
def clean_events(events, start_time=None, end_time=None, duration=None, pre=None, post=None):
    cleaned_events = clean_events_core(events,
                                       start_time=start_time,
                                       end_time=end_time,
                                       duration=duration,
                                       pre=pre,
                                       post=post)
    return cleaned_events


@task()
def concatenate_events_across_experiments(event_list):
    events = concatenate_events_across_experiments_core(event_list)
    return events


@task()
def select_word_events(all_events, encoding_only=True):
    all_events = select_word_events_core(all_events, encoding_only=encoding_only)
    return all_events
