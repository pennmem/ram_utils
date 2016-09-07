__author__ = 'm'

import os
import os.path
import numpy as np
from ptsa.data.readers import BaseEventReader

from RamPipeline import *
from ReportUtils import ReportRamTask


class FREventPreparation(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(FREventPreparation, self).__init__(mark_as_completed)
        self.params = params

    def run(self):
        try:

            events = None
            fr1_e_path = ''
            catfr1_e_path = ''

            if self.params.include_fr1:
                try:
                    fr1_e_path = os.path.join(self.pipeline.mount_point, 'data/events/RAM_FR1',
                                              self.pipeline.subject + '_events.mat')
                    e_reader = BaseEventReader(filename=fr1_e_path, eliminate_events_with_no_eeg=True)
                    events = e_reader.read()

                    # removing stim fileds that shouldn't be in non-stim experiments
                    evs_field_list = ['session','list','serialpos','type','item','itemno',
                                      'recalled','mstime','msoffset','rectime','intrusion',
                                      'eegoffset','eegfile'
                                     ]
                    events = events[evs_field_list].copy()

                    ev_order = np.argsort(events, order=('session', 'list', 'mstime'))
                    events = events[ev_order]
                except IOError:
                    pass

            if self.params.include_catfr1:
                try:
                    catfr1_e_path = os.path.join(self.pipeline.mount_point, 'data/events/RAM_CatFR1',
                                                 self.pipeline.subject + '_events.mat')
                    e_reader = BaseEventReader(filename=catfr1_e_path, eliminate_events_with_no_eeg=True)
                    catfr1_events = e_reader.read()

                    # removing stim fileds that shouldn't be in non-stim experiments
                    evs_field_list = ['session','list','serialpos','type','item','itemno',
                                      'recalled','mstime','msoffset','rectime','intrusion',
                                      'eegoffset','eegfile'
                                     ]
                    catfr1_events = catfr1_events[evs_field_list].copy()

                    ev_order = np.argsort(catfr1_events, order=('session', 'list', 'mstime'))
                    catfr1_events = catfr1_events[ev_order]
                    if events is None:
                        events = catfr1_events
                    else:
                        catfr1_events.session += 100
                        fields = list(set(events.dtype.names).intersection(catfr1_events.dtype.names))
                        events = np.hstack((events[fields], catfr1_events[fields])).view(np.recarray)
                except IOError:
                    pass

            events = events[events.type == 'WORD']

            print len(events), 'WORD events'

            self.pass_object('FR_events', events)


        except Exception:
            self.raise_and_log_report_exception(
                exception_type='MissingDataError',
                exception_message='Missing FR1 or CatFR1 events data (%s,%s)' % (fr1_e_path, catfr1_e_path)
            )
