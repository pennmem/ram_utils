__author__ = 'm'

import os

from ptsa.data.readers import TalReader

from ReportUtils import RamTask

class TalPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        super(TalPreparation,self).__init__(mark_as_completed)

    def run(self):
        tal_path = os.path.join(self.pipeline.mount_point,'data/eeg',self.pipeline.subject,'tal',self.pipeline.subject+'_talLocs_database_bipol.mat')
        tal_reader = TalReader(filename=tal_path)

        bpTalStruct = tal_reader.read()
        monopolar_channels = tal_reader.get_monopolar_channels()
        bipolar_pairs = tal_reader.get_bipolar_pairs()

        for i,bp in enumerate(bpTalStruct):
            bpTalStruct.tagName[i] = bp.tagName.upper()

        self.pass_object('monopolar_channels', monopolar_channels)
        self.pass_object('bipolar_pairs', bipolar_pairs)
        self.pass_object('bp_tal_structs', bpTalStruct)
