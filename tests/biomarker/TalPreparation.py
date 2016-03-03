__author__ = 'm'

import numpy as np
import os

from get_bipolar_subj_elecs import get_bipolar_subj_elecs

from RamPipeline import *


class TalPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        subjpath = os.path.join(self.pipeline.mount_point,'data/eeg',self.pipeline.subject)
        bipolar_pairs = get_bipolar_subj_elecs(subjpath, leadsonly=True, exclude_bad_leads=False)
        monopolar_channels = get_single_elecs_from_bps(bipolar_pairs)
        print len(monopolar_channels), 'single electrodes', len(bipolar_pairs), 'bipolar pairs'

        self.pass_object('bipolar_pairs', bipolar_pairs)
        self.pass_object('monopolar_channels', monopolar_channels)


def get_single_elecs_from_bps(bipolar_pairs):
    monopolar_channels = np.array([], dtype=np.dtype('|S32'))
    for ti in bipolar_pairs:
        monopolar_channels = np.hstack((monopolar_channels, ti['channel_str']))
    return np.unique(monopolar_channels)
