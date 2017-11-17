"""Partial pipelines for processing events that is used by full pipelines."""

from collections import namedtuple
from ramutils.events import *
from ramutils.tasks import task

__all__ = [
    'preprocess_fr_events',
    'combine_events',
]


ProcessedEvents = namedtuple("ProcessedEvents", "encoding, retrieval")


# FIXME: also return stim events?
@task()
def preprocess_fr_events(subject, root='/'):
    """Pre-processing for FR experiments.

    Parameters
    ----------
    subject : str
        Subject ID.
    root: str
        Base path for finding event files etc.

    Returns
    -------
    processed_events : ProcessedEvents
        Tuple containing encoding and retrieval events.


    """
    fr_events = load_events(subject, 'FR1', rootdir=root)
    catfr_events = load_events(subject, 'catFR1', rootdir=root)
    raw_events = concatenate_events_across_experiments([fr_events,
                                                        catfr_events])
    cleaned_events = clean_events(raw_events)
    all_events = insert_baseline_retrieval_events(cleaned_events, 1000, 29000)

    word_events = select_word_events(all_events, include_retrieval=True)
    encoding_events = select_encoding_events(word_events)
    retrieval_events = select_all_retrieval_events(word_events)

    processed_events = ProcessedEvents(encoding_events, retrieval_events)

    return processed_events


@task()
def combine_events(event_list):
    events = concatenate_events_for_single_experiment(event_list)
    return events
