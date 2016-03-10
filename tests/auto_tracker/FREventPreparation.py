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
        if self.dependency_inventory:
            self.dependency_inventory.add_dependent_resource(resource_name='FR1_events',
                                        access_path = ['experiments','FR1','events'])

            self.dependency_inventory.add_dependent_resource(resource_name='tal_bipolar',
                                        access_path = ['electrodes_info','tal_bipolar'])


    def run(self):
        events = None
        print 'GOT HERE'
        # if self.params.include_fr1:
        #     try:
        #         e_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_FR1', self.pipeline.subject + '_events.mat')
        #         e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
        #         events = e_reader.read()
        #         ev_order = np.argsort(events, order=('session','list','mstime'))
        #         events = events[ev_order]
        #     except IOError:
        #         pass
        #
        # if self.params.include_catfr1:
        #     try:
        #         e_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_CatFR1', self.pipeline.subject + '_events.mat')
        #         e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
        #         catfr1_events = e_reader.read()
        #         ev_order = np.argsort(catfr1_events, order=('session','list','mstime'))
        #         catfr1_events = catfr1_events[ev_order]
        #         if events is None:
        #             events = catfr1_events
        #         else:
        #             catfr1_events.session += 100
        #             fields = list(set(events.dtype.names).intersection(catfr1_events.dtype.names))
        #             events = np.hstack((events[fields],catfr1_events[fields])).view(np.recarray)
        #     except IOError:
        #         pass
        #
        # events = events[events.type == 'WORD']
        #
        # print len(events), 'WORD events'
        #
        # self.pass_object('FR_events', events)
