import numpy as np
from RamPipeline import *
from sklearn.externals import joblib


class ComputeModalControllability(RamTask):
    """ Using processed DTI data, calculate modal controllability for each electrode in a subject's montage """
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.controllability = None
   
    def restore(self):
        subject = self.pipeline.subject
        try:
            controllability_table = joblib.load(self.get_path_to_resource_in_workspace('-'.join([subject,'controllability.pkl'])))
            self.pass_object('controllability', controllability_table)
        except (IOError, AssertionError):
            self.run()
        return

    def run(self):
        subject = self.pipeline.subject

        bp_tal_structs = self.get_passed_object('bp_tal_structs')
        self.controllability_table = bp_tal_structs.reset_index()
        self.controllability_table = self.controllability_table[["index", "etype", "bp_atlas_loc"]]
        self.controllability_table = self.controllability_table.rename(columns={'index': 'Electrode Pair',
                                                                                'etype': 'Type',
                                                                                'bp_atlas_loc': 'Atlas Loc'})
        num_electrodes = len(self.controllability_table)
        fake_data = np.random.random(size=num_electrodes)
        self.controllability_table['Controllability'] = fake_data
        self.controllability_table = self.controllability_table.sort_values('Controllability', ascending=False)

        joblib.dump(self.controllability_table, self.get_path_to_resource_in_workspace('-'.join([subject, 'controllability.pkl'])))
        self.pass_object('controllability', self.controllability_table)
        return