"""Demo of what was formerly called 'FREventPreparation'."""

import os.path
from ptsa.data.readers import JsonIndexReader
from ramutils.tasks import memory
from ramutils.tasks.events import *

try:
    memory.clear(warn=False)
except:
    pass

jr = JsonIndexReader(os.path.expanduser('~/mnt/rhino/protocols/r1.json'))
subject = 'R1354E'

fr_events = read_fr_events(jr, subject, cat=False)
catfr_events = read_fr_events(jr, subject, cat=True)
all_events = concatenate_events(fr_events, catfr_events)
matched_events = create_baseline_events(all_events, 1000, 29000)
word_events = select_word_events(matched_events)

# Make sure to `conda install graphviz python-graphviz`
word_events.visualize()
