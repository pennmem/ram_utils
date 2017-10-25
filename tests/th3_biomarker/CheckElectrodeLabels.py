import sys
from ReportUtils import RamTask

class CheckElectrodeLabels(RamTask):
    def __init__(self, params, mark_as_completed=True):
        super(CheckElectrodeLabels,self).__init__(mark_as_completed)
        self.params = params

    def run(self):
        bp_tal_structs = self.get_passed_object('bp_tal_structs')

        anode_num = self.params.stim_params.elec1
        cathode_num = self.params.stim_params.elec2

        stim_pair = self.params.stim_params.anode + '-' + self.params.stim_params.cathode

        stim_pair_found = False
        print 'type(bp_tal_structs):',type(bp_tal_structs)
        print 'dtypes: ',bp_tal_structs.dtypes
        for tagName,row in bp_tal_structs.iterrows():
            if (int(row['channel_1']) == anode_num) and (int(row['channel_2']) == cathode_num):
                if tagName != stim_pair:
                    print 'Wrong stim pair for %d-%d: expected %s, found %s' % (anode_num,cathode_num,stim_pair,tagName)
                    sys.exit(1)
                stim_pair_found = True

        if not stim_pair_found:
            print 'WARNING: Stim pair was not found in bpTalStruct, assuming stim-only'
