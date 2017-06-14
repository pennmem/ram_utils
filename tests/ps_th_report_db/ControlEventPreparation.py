__author__ = 'm'

import os.path

from ReportUtils import ReportRamTask
from ptsa.data.readers import BaseEventReader

from ram_utils.RamPipeline import *


class ControlEventPreparation(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ControlEventPreparation,self).__init__(mark_as_completed)
        self.params = params

    def run(self):
        e_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_PS', self.pipeline.subject + '_events.mat')
        e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

        try:
            events = e_reader.read()
            events = events[events.type == 'SHAM']

        except Exception:
            # raise MissingDataError('Missing or Corrupt PS event file')

            self.raise_and_log_report_exception(
                                                exception_type='MissingDataError',
                                                exception_message='Missing or Corrupt PS event file'
                                                )

        print len(events), 'SHAM events'

        self.pass_object('control_events', events)
