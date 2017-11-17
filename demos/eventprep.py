"""Demo of what was formerly called 'FREventPreparation'."""

import os.path
import time
import functools
from contextlib import contextmanager
from pkg_resources import resource_filename

from ramutils.tasks import memory
from ramutils.events import *


datafile = functools.partial(resource_filename, 'ramutils.test.test_data')

@contextmanager
def timeit():
    t0 = time.time()
    yield
    dt = time.time() - t0
    print("Completed in {} s".format(dt))


try:
    memory.clear(warn=False)
except:
    pass

subject = 'R1354E'
fr_events = load_events(subject, 'FR1')
catfr_events = load_events(subject, 'catFR1')
all_events = concatenate_events([fr_events, catfr_events])
matched_events = insert_baseline_retrieval_events(all_events, 1000, 29000)
word_events = select_word_events(matched_events)

# Make sure to `conda install gaphviz python-graphviz`
word_events.visualize()

with timeit():
    output = word_events.compute()
    print(len(output), "events")
