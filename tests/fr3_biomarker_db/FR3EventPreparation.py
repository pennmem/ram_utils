__author__ = 'm'

import os.path

import numpy as np
from ptsa.data.readers import BaseEventReader

from ram_utils.RamPipeline import *


def construct_stim_item_mask(events):
    word_event_mask = (events.type == 'PRACTICE_WORD') | (events.type == 'WORD')
    n_word_events = np.sum(word_event_mask)
    stim_item_mask = np.zeros(n_word_events, dtype=np.bool)
    j = 0
    for i in xrange(len(events)):
        if word_event_mask[i]:
            stim_item_mask[j] = (events[i+1].isStim==1 or events[i+2].isStim==1)
            j += 1
    return stim_item_mask


class FR3EventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        task3 = 'RAM_FR3'
        e_path = os.path.join(self.pipeline.mount_point , 'data/events', task3, self.pipeline.subject + '_events.mat')
        e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
        all_events = e_reader.read()

        ev_order = np.argsort(all_events, order=('session','list','mstime'))
        all_events = all_events[ev_order]

        events = all_events[(all_events.type == 'PRACTICE_WORD') | (all_events.type == 'WORD')]

        print len(events), task3, 'WORD events'

        stim_item_mask = construct_stim_item_mask(all_events)

        self.pass_object(task3+'_events', events)
        self.pass_object(task3+'_all_events', all_events)
        self.pass_object(task3+'_stim_item_mask', stim_item_mask)
