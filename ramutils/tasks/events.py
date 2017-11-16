"""Partial pipelines for processing events that is used by full pipelines."""

from collections import namedtuple
from ramutils.events import *
from ramutils.tasks import task

__all__ = [
    'preprocess_fr_events'
]


ProcessedEvents = namedtuple("ProcessedEvents", "encoding, retrieval")


# FIXME: also return stim events?
@task()
def preprocess_fr_events(subject):
    """Pre-processing for FR experiments.

    Parameters
    ----------
    subject : str
        Subject ID.

    Returns
    -------
    processed_events : ProcessedEvents
        Tuple containing encoding and retrieval events.


    """
    fr_events = load_events(subject, 'FR1')
    catfr_events = load_events(subject, 'catFR1')
    raw_events = concatenate_events_across_experiments([fr_events,
                                                        catfr_events])
    cleaned_events = clean_events(raw_events)
    all_events = insert_baseline_retrieval_events(cleaned_events, 1000, 29000)

    word_events = select_word_events(all_events, include_retrieval=True)
    encoding_events = select_encoding_events(word_events)
    retrieval_events = select_retrieval_events(word_events)

    processed_events = ProcessedEvents(encoding_events, retrieval_events)

    return processed_events
