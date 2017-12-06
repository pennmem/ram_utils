"""Partial pipelines for processing events that is used by full pipelines."""

from ramutils.events import preprocess_events as preprocess_events_core
from ramutils.events import select_word_events as select_word_events_core
from ramutils.tasks import task

__all__ = [
    'preprocess_events',
    'select_word_events'
]


@task()
def preprocess_events(subject, experiment, start_time, end_time, duration, pre, post, sessions=None,
                      combine_events=True, root='/'):
    processed_events = preprocess_events_core(subject,
                                              experiment,
                                              start_time,
                                              end_time,
                                              duration,
                                              pre,
                                              post,
                                              sessions=sessions,
                                              combine_events=combine_events,
                                              root=root)
    return processed_events


@task()
def select_word_events(all_events, encoding_only=True):
    all_events = select_word_events_core(all_events, encoding_only=encoding_only)
    return all_events
