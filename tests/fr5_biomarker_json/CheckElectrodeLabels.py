import sys
from ReportUtils import RamTask

class CheckElectrodeLabels(RamTask):
    def __init__(self, params, mark_as_completed=True):
        super(CheckElectrodeLabels,self).__init__(mark_as_completed)
        self.params = params

    def run(self):
        bp_tal_structs = self.get_passed_object('bp_tal_structs')

        for (anode,anode_num,cathode,cathode_num) in zip(self.params.stim_params.anodes,self.params.stim_params.anode_nums,
                                                         self.params.stim_params.cathodes,self.params.stim_params.cathode_nums):

            stim_pair = anode + '-' + cathode

            if not stim_pair in bp_tal_structs.index:
                print 'Unknown bipolar pair:', stim_pair
                sys.exit(1)

            found_anode_num = int(bp_tal_structs.channel_1[stim_pair])
            found_cathode_num = int(bp_tal_structs.channel_2[stim_pair])

            if (anode_num != found_anode_num) or (cathode_num != found_cathode_num):
                print 'Wrong stim pair for %s: expected %d-%d, found %d-%d' % (stim_pair,anode_num,cathode_num,found_anode_num,found_cathode_num)
                sys.exit(1)
