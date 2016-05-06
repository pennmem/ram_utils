__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader

from RamPipeline import *


class FREventPreparation(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params

    def run(self):
        events = None
        if self.params.include_fr1:
            try:
                e_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_FR1', self.pipeline.subject + '_events.mat')
                e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
                events = e_reader.read()
                ev_order = np.argsort(events, order=('session','list','mstime'))
                events = events[ev_order]
                events = events[events.type == 'WORD']
            except IOError:
                print 'No FR1 events found'

        if self.params.include_catfr1:
            try:
                e_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_CatFR1', self.pipeline.subject + '_events.mat')
                e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
                catfr1_events = e_reader.read()
                ev_order = np.argsort(catfr1_events, order=('session','list','mstime'))
                catfr1_events = catfr1_events[ev_order]
                catfr1_events = catfr1_events[catfr1_events.type == 'WORD']
                if events is None:
                    events = catfr1_events
                else:
                    catfr1_events.session += 100
                    fields = list(set(events.dtype.names).intersection(catfr1_events.dtype.names))
                    events = np.hstack((events[fields],catfr1_events[fields])).view(np.recarray)
            except IOError:
                print 'No CatFR1 events found'

        if self.params.include_fr3:
            try:
                e_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_FR3', self.pipeline.subject + '_events.mat')
                e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
                fr3_events = e_reader.read()
                ev_order = np.argsort(fr3_events, order=('session','list','mstime'))
                fr3_events = fr3_events[ev_order]
                fr3_events = fr3_events[(fr3_events.type == 'WORD') & (fr3_events.stimList == 0)]
                fr3_events.session += 200
                fields = list(set(events.dtype.names).intersection(fr3_events.dtype.names))
                events = np.hstack((events[fields],fr3_events[fields])).view(np.recarray)
            except IOError:
                print 'No FR3 events found'

        if self.params.include_catfr3:
            try:
                e_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_CatFR3', self.pipeline.subject + '_events.mat')
                e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
                catfr3_events = e_reader.read()
                ev_order = np.argsort(catfr3_events, order=('session','list','mstime'))
                catfr3_events = catfr3_events[ev_order]
                catfr3_events = catfr3_events[(catfr3_events.type == 'WORD') & (catfr3_events.stimList == 0)]
                catfr3_events.session += 300
                fields = list(set(events.dtype.names).intersection(catfr3_events.dtype.names))
                events = np.hstack((events[fields],catfr3_events[fields])).view(np.recarray)
            except IOError:
                print 'No CatFR3 events found'

        print len(events), 'WORD events in total'

        self.pass_object('FR_events', events)
