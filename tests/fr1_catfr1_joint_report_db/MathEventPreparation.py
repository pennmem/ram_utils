__author__ = 'm'

import os.path

import numpy as np
from ReportUtils import ReportRamTask
from ptsa.data.readers import BaseEventReader

from ram_utils.RamPipeline import *


class MathEventPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(MathEventPreparation,self).__init__(mark_as_completed)

    def run(self):
        events = None
        fr1_e_path=''
        catfr1_e_path = ''

        try:
            try:
                fr1_e_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_FR1', self.pipeline.subject + '_math.mat')
                e_reader = BaseEventReader(filename=fr1_e_path, eliminate_events_with_no_eeg=True)
                events = e_reader.read()
                print "Got FR1 math events"
            except IOError:
                pass

            try:
                catfr1_e_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_CatFR1', self.pipeline.subject + '_math.mat')
                e_reader = BaseEventReader(filename=catfr1_e_path, eliminate_events_with_no_eeg=True)
                catfr1_events = e_reader.read()
                print "Got CatFR1 math events"
                if events is None:
                    events = catfr1_events
                else:
                    print "Joining FR1 and CatFR1"
                    catfr1_events.session = -catfr1_events.session-1
                    fields = list(set(events.dtype.names).intersection(catfr1_events.dtype.names))
                    events = np.hstack((events[fields],catfr1_events[fields])).view(np.recarray)
            except IOError:
                pass

            if events is not None:
                events = events[events.type == 'PROB']
                print len(events), 'PROB events'

            self.pass_object(self.pipeline.task+'_math_events', events)

        except Exception:
            self.raise_and_log_report_exception(
                                                exception_type='MissingDataError',
                                                exception_message='Missing FR1 or CatFR1 events data (%s,%s)'%(fr1_e_path,catfr1_e_path)
                                                )
