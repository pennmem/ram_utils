__author__ = 'm'

import numpy as np
import os

from get_bipolar_subj_elecs import get_bipolar_subj_elecs

from sklearn.externals import joblib

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

        for i,bp in enumerate(bipolar_pairs):
            bipolar_pairs.tagName[i] = bp.tagName.upper()

        loc_tag = dict(zip(bipolar_pairs.tagName, bipolar_pairs.locTag))

        try:
            stim_bipolar_pairs = get_bipolar_subj_elecs(subjpath, leadsonly=True, exclude_bad_leads=False, bipolfileend='_talLocs_database_stimOnly.mat')

            for i,bp in enumerate(stim_bipolar_pairs):
                stim_bipolar_pairs.tagName[i] = bp.tagName.upper()

            loc_tag.update(dict(zip(stim_bipolar_pairs.tagName, stim_bipolar_pairs.locTag)))
        except:
            pass

        self.pass_object('loc_tag', loc_tag)
        joblib.dump(loc_tag, self.get_path_to_resource_in_workspace(self.pipeline.subject+'-loc_tag.pkl'))


def get_single_elecs_from_bps(bipolar_pairs):
    monopolar_channels = np.array([], dtype=np.dtype('|S32'))
    for ti in bipolar_pairs:
        monopolar_channels = np.hstack((monopolar_channels, ti['channel_str']))
    return np.unique(monopolar_channels)
