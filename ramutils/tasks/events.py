"""Partial pipelines for processing events that is used by full pipelines."""

from collections import namedtuple
from ramutils.tasks import *
from ramutils.events import *


ProcessedEvents = namedtuple("ProcessedEvents", "encoding, retrieval")


# FIXME: also return stim events?
def preprocess_fr_events(index, subject, compute=False):
    """Pre-processing for FR experiments.

    Parameters
    ----------
    index : JsonIndexReader
        Used to find paths to stored events.
    subject : str
        Subject ID.
    compute : bool
        Compute the task graph before returning (default: False).

    Returns
    -------
    processed_events : ProcessedEvents
        Tuple containing encoding and retrieval events.


    """
    fr_events = read_fr_events(index, subject, cat=False)
    catfr_events = read_fr_events(index, subject, cat=True)
    raw_events = concatenate_events(fr_events)
    all_events = create_baseline_events(raw_events, 1000, 29000)
    word_events = select_word_events(all_events, include_retrieval=True)
    encoding_events = select_encoding_events(word_events)
    retrieval_events = select_retrieval_events(word_events)

    if compute:
        processed_events = ProcessedEvents(encoding_events.compute(), retrieval_events.compute())
    else:
        processed_events = ProcessedEvents(encoding_events, retrieval_events)

    return processed_events
