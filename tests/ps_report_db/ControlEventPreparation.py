__author__ = 'm'

import os
import os.path
import numpy as np
from ptsa.data.readers import BaseEventReader

from ram_utils.RamPipeline import *
from ReportUtils import ReportRamTask
from ReportUtils import ReportRamTask

class ControlEventPreparation(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ControlEventPreparation,self).__init__(mark_as_completed)
        self.params = params

    def run(self):
        e_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_PS', self.pipeline.subject + '_events.mat')
        e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

        try:
            events = e_reader.read()
            ev_order = np.argsort(events, order=('session','mstime'))
            events = events[ev_order]

            # try:
            #     events = Events(get_events(subject=subject, task='RAM_PS', path_prefix=self.pipeline.mount_point))
            # except IOError:
            #     raise Exception('No parameter search for subject %s' % subject)
            #

            events = events[events.type == 'SHAM']

        except Exception:
            # raise MissingDataError('Missing or Corrupt PS event file')

            raise
            self.raise_and_log_report_exception(
                                                exception_type='MissingDataError',
                                                exception_message='Missing or Corrupt PS event file'
                                                )

        print len(events), 'SHAM events'

        self.pass_object('control_events', events)
