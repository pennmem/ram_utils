__author__ = 'm'

import numpy as np
import os

from get_bipolar_subj_elecs import get_bipolar_subj_elecs

from RamPipeline import *

from sklearn.externals import joblib


class TalPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):

        subject = self.pipeline.subject
        task = self.pipeline.task
        events = self.get_passed_object(task+'_events')


        from ptsa.data.readers.TalReader import TalReader
        tal_path = os.path.join(self.pipeline.mount_point,'data/eeg',subject,'tal',subject+'_talLocs_database_bipol.mat')

        tal_reader = TalReader(tal_filename=tal_path)
        monopolar_channels = tal_reader.get_monopolar_channels()
        bipolar_pairs = tal_reader.get_bipolar_pairs()




        # tal_info = get_bps(events)
        # channels = get_single_elecs_from_bps(tal_info)
        # loc_info = dict(zip(tal_info.tagName, tal_info.locTag))

        print len(monopolar_channels), 'single electrodes', len(bipolar_pairs), 'bipolar pairs'

        self.pass_object('bipolar_pairs', bipolar_pairs)
        self.pass_object('monopolar_channels', monopolar_channels)
        # self.pass_object('loc_info', loc_info)

        # joblib.dump(loc_info, self.get_path_to_resource_in_workspace(subject+'-tag_info.pkl'))

# def get_bps(events):
#     dataroot = get_dataroot(events)
#     subjpath = os.path.dirname(os.path.dirname(dataroot))
#     return get_bipolar_subj_elecs(subjpath, leadsonly=True, exclude_bad_leads=False)
#
#
# def get_single_elecs_from_bps(tal_info):
#     channels = np.array([], dtype=np.dtype('|S32'))
#     for ti in tal_info:
#         channels = np.hstack((channels, ti['channel_str']))
#     return np.unique(channels)
#
#
# def get_dataroot(events):
#     dataroots = np.unique([esrc.dataroot for esrc in events.esrc])
#     return dataroots[0]
