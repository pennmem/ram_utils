__author__ = 'm'

import numpy as np
import os


from get_bipolar_subj_elecs import get_bipolar_subj_elecs


from RamPipeline import *

class TalPreparation(RamTask):
    def __init__(self, task, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.task=task

    def run(self):
        events = self.get_passed_object(self.task+'_events')
        tal_info = get_bps(events)
        channels = get_single_elecs_from_bps(tal_info)
        print len(channels), 'single electrodes', len(tal_info), 'bipolar pairs'

        self.pass_object('tal_info',tal_info)
        self.pass_object('channels',channels)



def get_bps(events):
    dataroot = get_dataroot(events)
    subjpath = os.path.dirname(os.path.dirname(dataroot))
    return get_bipolar_subj_elecs(subjpath, leadsonly=True, exclude_bad_leads=False)


def get_single_elecs_from_bps(tal_info):
    channels = np.array([], dtype=np.dtype('|S32'))
    for ti in tal_info:
        channels = np.hstack((channels, ti['channel_str']))
    return np.unique(channels)


def get_dataroot(events):
    dataroots = np.unique([esrc.dataroot for esrc in events.esrc])
    if len(dataroots) != 1:
        raise ValueError('Invalid number of dataroots: %d' % len(dataroots))
    return dataroots[0]