__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader

from RamPipeline import *


class FREventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        task = self.pipeline.task

        e_path = os.path.join(self.pipeline.mount_point , 'data/events', task, self.pipeline.subject + '_events.mat')
        e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

        events = e_reader.read()
        events = events[events.type == 'WORD']
        ev_order = np.argsort(events, order=('session','list','mstime'))
        events = events[ev_order]

        print len(events), task, 'WORD events'

        task3 = task.replace('FR1','FR3')

        try:
            e_path = os.path.join(self.pipeline.mount_point , 'data/events', task3, self.pipeline.subject + '_events.mat')
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
            events3 = e_reader.read()

            events3 = events3[(events3.type == 'WORD') & (events3.stimList == 0)]
            ev_order = np.argsort(events3, order=('session','list','mstime'))
            events3 = events3[ev_order]

            events3.session += 100
            fields = list(set(events.dtype.names).intersection(events3.dtype.names))
            events = np.hstack((events[fields],events3[fields])).view(np.recarray)
        except IOError:
            pass

        print len(events), 'WORD events in total'

        self.pass_object('FR_events', events)
