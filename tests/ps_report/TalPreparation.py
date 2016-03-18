__author__ = 'm'

import numpy as np
import os

from ptsa.data.readers import TalReader, TalStimOnlyReader

from get_bipolar_subj_elecs import get_bipolar_subj_elecs

from sklearn.externals import joblib

from RamPipeline import *
from ReportUtils import MissingDataError,ReportRamTask


class TalPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        try:

            tal_path = os.path.join(self.pipeline.mount_point,'data/eeg',self.pipeline.subject,'tal',self.pipeline.subject+'_talLocs_database_bipol.mat')
            tal_stim_only_path = os.path.join(self.pipeline.mount_point,'data/eeg',self.pipeline.subject,'tal',self.pipeline.subject+'_talLocs_database_stimOnly.mat')
            tal_reader = TalReader(filename=tal_path)
            tal_stim_only_reader = TalStimOnlyReader(filename=tal_stim_only_path)

            bpTalStruct = tal_reader.read()
            monopolar_channels = tal_reader.get_monopolar_channels()
            bipolar_pairs = tal_reader.get_bipolar_pairs()

            for i,bp in enumerate(bpTalStruct):
                bpTalStruct.tagName[i] = bp.tagName.upper()

            loc_tag = dict()
            try:
                loc_tag = dict(zip(bpTalStruct.tagName, bpTalStruct.locTag))
            except AttributeError:
                pass

            # self.pass_object('bipolar_pairs', bipolar_pairs)
            self.pass_object('monopolar_channels', monopolar_channels)
            self.pass_object('bipolar_pairs', bipolar_pairs)


            try:
                virtualTalStruct = tal_stim_only_reader.read()
                for i,bp in enumerate(virtualTalStruct):
                    virtualTalStruct.tagName[i] = bp.tagName.upper()

                loc_tag.update(dict(zip(virtualTalStruct.tagName, virtualTalStruct.locTag)))

            except IOError:
                    pass

            self.pass_object('loc_tag', loc_tag)
            joblib.dump(loc_tag, self.get_path_to_resource_in_workspace(self.pipeline.subject+'-loc_tag.pkl'))

            self.add_report_status(message='OK')

        except Exception:
            raise MissingDataError('Missing or corrupt electrodes data %s for subject %s '%(tal_path,self.pipeline.subject))





        #-----------------------------


        # subjpath = os.path.join(self.pipeline.mount_point,'data/eeg',self.pipeline.subject)
        # bipolar_pairs = get_bipolar_subj_elecs(subjpath, leadsonly=True, exclude_bad_leads=False)
        # monopolar_channels = get_single_elecs_from_bps(bipolar_pairs)
        # print len(monopolar_channels), 'single electrodes', len(bipolar_pairs), 'bipolar pairs'
        #
        # self.pass_object('bipolar_pairs', bipolar_pairs)
        # self.pass_object('monopolar_channels', monopolar_channels)
        #
        # for i,bp in enumerate(bipolar_pairs):
        #     bipolar_pairs.tagName[i] = bp.tagName.upper()
        #
        # loc_tag = dict(zip(bipolar_pairs.tagName, bipolar_pairs.locTag))
        #
        # try:
        #     stim_bipolar_pairs = get_bipolar_subj_elecs(subjpath, leadsonly=True, exclude_bad_leads=False, bipolfileend='_talLocs_database_stimOnly.mat')
        #
        #     for i,bp in enumerate(stim_bipolar_pairs):
        #         stim_bipolar_pairs.tagName[i] = bp.tagName.upper()
        #
        #     loc_tag.update(dict(zip(stim_bipolar_pairs.tagName, stim_bipolar_pairs.locTag)))
        # except:
        #     pass
        #
        # self.pass_object('loc_tag', loc_tag)
        # joblib.dump(loc_tag, self.get_path_to_resource_in_workspace(self.pipeline.subject+'-loc_tag.pkl'))



def get_single_elecs_from_bps(bipolar_pairs):
    monopolar_channels = np.array([], dtype=np.dtype('|S32'))
    for ti in bipolar_pairs:
        monopolar_channels = np.hstack((monopolar_channels, ti['channel_str']))
    return np.unique(monopolar_channels)
