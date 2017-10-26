import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader

from ReportUtils import ReportRamTask


class FR1EventPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(FR1EventPreparation,self).__init__(mark_as_completed)

    def run(self):
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

        events = events[events.type == 'WORD']

        sessions = np.unique(events.session)
        print 'Sessions', sessions
        if len(sessions) < 3:
            raise Exception("Too few sessions")

        print self.pipeline.subject, 'has', len(events), 'WORD events'

        self.pass_object('events', events)
