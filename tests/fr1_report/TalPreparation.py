__author__ = 'm'

import numpy as np
import os

from get_bipolar_subj_elecs import get_bipolar_subj_elecs

from RamPipeline import *

from ReportUtils import ReportRamTask

class TalPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(TalPreparation,self).__init__(mark_as_completed)

    def run(self):
        events = self.get_passed_object(self.pipeline.task+'_events')
        bipolar_pairs = self.get_bps(events)
        monopolar_channels = get_single_elecs_from_bps(bipolar_pairs)
        print len(monopolar_channels), 'single electrodes', len(bipolar_pairs), 'bipolar pairs'

        self.pass_object('bipolar_pairs', bipolar_pairs)
        self.pass_object('monopolar_channels', monopolar_channels)


    def get_bps(self, events):
        #dataroot = get_dataroot(events)
        #subjpath = os.path.dirname(os.path.dirname(dataroot))
        subjpath = os.path.join(self.pipeline.mount_point,'data/eeg',self.pipeline.subject)
        return get_bipolar_subj_elecs(subjpath, leadsonly=True, exclude_bad_leads=False)


def get_single_elecs_from_bps(bipolar_pairs):
    monopolar_channels = np.array([], dtype=np.dtype('|S32'))
    for ti in bipolar_pairs:
        monopolar_channels = np.hstack((monopolar_channels, ti['channel_str']))
    return np.unique(monopolar_channels)


#def get_dataroot(events):
#    dataroots = np.unique([esrc.dataroot for esrc in events.esrc])
#    return dataroots[0]
