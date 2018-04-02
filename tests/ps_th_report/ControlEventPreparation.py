import os
import os.path
import numpy as np
from ptsa.data.readers import BaseEventReader

from ReportUtils import ReportRamTask
from ptsa.data.readers  import JsonIndexReader

class ControlEventPreparation(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ControlEventPreparation,self).__init__(mark_as_completed)
        self.params = params

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))
        event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage,
                                                               experiment=task)))
        events = None

        try:
            for sess_file in event_files:
                e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
                e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
                sess_events = e_reader.read()
                events = sess_events if events is None else np.hstack((events, sess_events))
            ev_order = np.argsort(events, order=('session','mstime'))
            events = events[ev_order]

            events = events[events.type == 'SHAM']

        except Exception:
            # raise MissingDataError('Missing or Corrupt PS event file')

            self.raise_and_log_report_exception(
                                                exception_type='MissingDataError',
                                                exception_message='Missing or Corrupt PS event file'
                                                )

        print len(events), 'SHAM events'

        self.pass_object('control_events', events)
