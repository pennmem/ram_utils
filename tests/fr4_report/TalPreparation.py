__author__ = 'm'

import numpy as np
import os

from ptsa.data.readers import TalReader, TalStimOnlyReader

from sklearn.externals import joblib

from ReportUtils import ReportRamTask


class TalPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(TalPreparation,self).__init__(mark_as_completed)

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
            self.raise_and_log_report_exception(
                                                exception_type='MissingDataError',
                                                exception_message='Missing or corrupt electrodes data %s for subject %s '%(tal_path,self.pipeline.subject)
                                                )


def get_single_elecs_from_bps(bipolar_pairs):
    monopolar_channels = np.array([], dtype=np.dtype('|S32'))
    for ti in bipolar_pairs:
        monopolar_channels = np.hstack((monopolar_channels, ti['channel_str']))
    return np.unique(monopolar_channels)
