__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader

from RamPipeline import *
from ReportUtils import MissingDataError


class FR1EventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        try:
            events = None
            try:
                fr1_e_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_FR1', self.pipeline.subject + '_events.mat')
                e_reader = BaseEventReader(filename=fr1_e_path, eliminate_events_with_no_eeg=True)
                events = e_reader.read()
                ev_order = np.argsort(events, order=('session','list','mstime'))
                events = events[ev_order]
            except IOError:
                pass

            try:
                catfr1_e_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_CatFR1', self.pipeline.subject + '_events.mat')
                e_reader = BaseEventReader(filename=catfr1_e_path, eliminate_events_with_no_eeg=True)
                catfr1_events = e_reader.read()
                ev_order = np.argsort(catfr1_events, order=('session','list','mstime'))
                catfr1_events = catfr1_events[ev_order]
                if events is None:
                    events = catfr1_events
                else:
                    catfr1_events.session += 100
                    fields = list(set(events.dtype.names).intersection(catfr1_events.dtype.names))
                    events = np.hstack((events[fields],catfr1_events[fields])).view(np.recarray)
            except IOError:
                pass

            self.pass_object('RAM_FR1_all_events', events)

            intr_events = events[(events.intrusion!=-999) & (events.intrusion!=0)]

            rec_events = events[events.type == 'REC_WORD']

            events = events[events.type == 'WORD']

            print len(events), 'WORD events'

            self.pass_object('RAM_FR1_events', events)
            self.pass_object('RAM_FR1_intr_events', intr_events)
            self.pass_object('RAM_FR1_rec_events', rec_events)

        except Exception:
            raise MissingDataError('Missing FR1 or CatFR1 data (%s,%s) for subject %s '%(fr1_e_path,catfr1_e_path,self.pipeline.subject))


        # task = self.pipeline.task
        #
        # e_path = os.path.join(self.pipeline.mount_point , 'data', 'events', task, self.pipeline.subject + '_events.mat')
        # e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
        #
        # events = e_reader.read()
        # ev_order = np.argsort(events, order=('session','list','mstime'))
        # events = events[ev_order]
        #
        # self.pass_object(self.pipeline.task+'_all_events', events)
        #
        # intr_events = events[(events.intrusion!=-999) & (events.intrusion!=0)]
        #
        # rec_events = events[events.type == 'REC_WORD']
        #
        # events = events[events.type == 'WORD']
        #
        # print len(events), task, 'WORD events'
        #
        # self.pass_object(task+'_events', events)
        # self.pass_object(self.pipeline.task+'_intr_events', intr_events)
        # self.pass_object(self.pipeline.task+'_rec_events', rec_events)
