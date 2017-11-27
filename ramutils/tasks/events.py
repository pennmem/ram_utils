"""Partial pipelines for processing events that is used by full pipelines."""

from ramutils.events import preprocess_events as preprocess_events_core
from ramutils.tasks import task

__all__ = [
    'preprocess_events',
]


@task()
def preprocess_events(subject, experiment, start_time, end_time, duration,
                      pre, post, combine_events=True, encoding_only=False,
                      root='/'):
    processed_events = preprocess_events_core(subject,
                                              experiment,
                                              start_time,
                                              end_time,
                                              duration,
                                              pre,
                                              post,
                                              combine_events=combine_events,
                                              encoding_only=encoding_only,
                                              root=root)
    return processed_events
