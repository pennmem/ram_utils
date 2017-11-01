"""Demo of what was formerly called 'FREventPreparation'."""

import os.path
import time
from contextlib import contextmanager

from ramutils.tasks import memory, read_index
from ramutils.tasks.events import *


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

jr = read_index(os.path.expanduser('~/mnt/rhino'))
fr_events = read_fr_events(jr, subject, cat=False)
catfr_events = read_fr_events(jr, subject, cat=True)
all_events = concatenate_events(fr_events, catfr_events)
matched_events = create_baseline_events(all_events, 1000, 29000)
word_events = select_word_events(matched_events)

# Make sure to `conda install graphviz python-graphviz`
word_events.visualize()

with timeit():
    output = word_events.compute()
    print(output)
