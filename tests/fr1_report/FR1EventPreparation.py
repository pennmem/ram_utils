__author__ = 'm'

import os
import os.path
import re
import numpy as np
from scipy.io import loadmat

from ptsa.data.readers import BaseEventReader

from RamPipeline import *


class FR1EventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        task = self.pipeline.task

        e_path = os.path.join(self.pipeline.mount_point , 'data', 'events', task, self.pipeline.subject + '_events.mat')
        e_reader = BaseEventReader(event_file=e_path, eliminate_events_with_no_eeg=True, use_ptsa_events_class=False)

        events = e_reader.read()

        self.pass_object(self.pipeline.task+'_all_events', events)

        # events = Events(get_events(subject=self.pipeline.subject, task=task, path_prefix=self.pipeline.mount_point))

        # events = correct_eegfile_field(events)
        # ev_order = np.argsort(events, order=('session','list','mstime'))
        # events = events[ev_order]
        #
        # events = self.attach_raw_bin_wrappers(events)

        intr_events = events[(events.intrusion!=-999) & (events.intrusion!=0)]

        rec_events = events[events.type == 'REC_WORD']

        events = events[events.type == 'WORD']

        print len(events), task, 'WORD events'

        self.pass_object(task+'_events', events)
        self.pass_object(self.pipeline.task+'_intr_events', intr_events)
        self.pass_object(self.pipeline.task+'_rec_events', rec_events)
