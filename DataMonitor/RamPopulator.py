import os
from os.path import *
from JSONUtils import JSONNode

from key_utils import compute_md5_key

class RamPopulator(object):
    def __init__(self):
        self.mount_point = '/Users/m/'
        self.subject_dir_target = join(self.mount_point,'data/subjects')
        self.subject_dir_source = join(self.mount_point,'data/eeg')
        self.protocol='R1'
        self.version = '1'

    def get_list_of_subjects(self,protocol):
        subjects = [
            d for d in os.listdir(self.subject_dir_source)
            if isdir(join(self.subject_dir_source, d))
            and d.startswith(protocol)
                    ]
        return subjects

    def create_subject_info_dir(self,path):
        subjects = self.get_list_of_subjects(protocol=self.protocol)
        for subject in subjects:
            try:
                os.makedirs(join(path,subject))
            except OSError:
                # in case directory exists
                pass

    def create_experiment_JSON_stub(self,subject_code, experiment_name,prefix, description):
        node = JSONNode()

        p = join(prefix,subject_code+'_events.mat')
        self.attach_single_file_JSON_stub(parent_node=node,
                                          json_stub_name='events',
                                          full_path=join(self.mount_point,'data', p),
                                          partial_path=p)

        p = join(prefix,subject_code+'_math.mat')
        self.attach_single_file_JSON_stub(parent_node=node,
                                          json_stub_name='math_events',
                                          full_path=join(self.mount_point,'data', p),
                                          partial_path=p)


        p = join(prefix,subject_code+'_expinfo.mat')
        self.attach_single_file_JSON_stub(parent_node=node,
                                          json_stub_name='info',
                                          full_path=join(self.mount_point,'data', p),
                                          partial_path=p)


        node['description'] = description

        # return experiment_node
        return node


    def attach_single_file_JSON_stub(self, parent_node, json_stub_name, full_path, partial_path=''):
        node = JSONNode()
        if not isfile(full_path):
            return

        node['path'] = partial_path if partial_path else full_path
        node['md5'] = compute_md5_key(full_path)

        parent_node.add_child_node(json_stub_name,node)



    def create_subject_JSON_stub(self,subject_code):
        root_node = JSONNode()
        root_node['version'] = self.version
        subject_node = root_node.add_child_node('subject')
        subject_node['code'] = subject_code

        # ------------------- electrodes ---------------------------------
        electrodes_info = root_node.add_child_node('electrodes_info')

        p = join('eeg',subject_code,'tal',subject_code+'_talLocs_database_bipol.mat')
        self.attach_single_file_JSON_stub(parent_node=electrodes_info,
                                          json_stub_name='tal_bipolar',
                                          full_path=join(self.mount_point,'data', p),
                                          partial_path=p)


        p = join('eeg',subject_code,'tal',subject_code+'_talLocs_database_monopol.mat')
        self.attach_single_file_JSON_stub(parent_node=electrodes_info,
                                          json_stub_name='tal_monopolar',
                                          full_path=join(self.mount_point,'data', p),
                                          partial_path=p)


        # --------------------- eeg ---------------------------------
        eeg_node = root_node.add_child_node('eeg')
        eeg_reref_node = eeg_node.add_child_node('reref_dir')
        eeg_reref_node['path'] = join('eeg',subject_code,'eeg.reref')

        eeg_noreref_node = eeg_node.add_child_node('noreref_dir')
        eeg_noreref_node['path'] = join('eeg',subject_code,'eeg.noreref')


        p = join('eeg',subject_code,'eeg.reref','params.txt')
        self.attach_single_file_JSON_stub(parent_node=eeg_node,
                                          json_stub_name='params_reref',
                                          full_path=join(self.mount_point,'data', p),
                                          partial_path=p)

        p = join('eeg',subject_code,'eeg.noreref','params.txt')
        self.attach_single_file_JSON_stub(parent_node=eeg_node,
                                          json_stub_name='params_noreref',
                                          full_path=join(self.mount_point,'data', p),
                                          partial_path=p)



        # --------------------- experiments ---------------------------------
        experiments_node = root_node.add_child_node('experiments')

        fr1_node = self.create_experiment_JSON_stub(
            subject_code=subject_code,
            experiment_name='FR1',
            prefix='events/RAM_FR1',
            description='Free Recall - record-only experiment'
        )

        experiments_node.add_child_node('FR1',fr1_node)

        fr2_node = self.create_experiment_JSON_stub(
            subject_code=subject_code,
            experiment_name='FR2',
            prefix='events/RAM_FR2',
            description='Free Recall - open-loop stimulation  experiment'
        )

        experiments_node.add_child_node('FR2',fr2_node)

        fr3_node = self.create_experiment_JSON_stub(
            subject_code=subject_code,
            experiment_name='FR3',
            prefix='events/RAM_FR3',
            description='Free Recall - closed-loop stimulation  experiment'
        )

        experiments_node.add_child_node('FR3',fr3_node)

        catfr1_node = self.create_experiment_JSON_stub(
            subject_code=subject_code,
            experiment_name='CatFR1',
            prefix='events/RAM_CatFR1',
            description='Categorized Free Recall - record-only experiment'
        )

        experiments_node.add_child_node('CatFR1',catfr1_node)

        catfr2_node = self.create_experiment_JSON_stub(
            subject_code=subject_code,
            experiment_name='CatFR2',
            prefix='events/RAM_CatFR2',
            description='Categorized Free Recall - open-loop stimulation  experiment'
        )

        experiments_node.add_child_node('CatFR2',catfr1_node)

        catfr3_node = self.create_experiment_JSON_stub(
            subject_code=subject_code,
            experiment_name='CatFR3',
            prefix='events/RAM_CatFR3',
            description='Categorized Free Recall - closed-loop stimulation  experiment'
        )

        experiments_node.add_child_node('CatFR3',catfr3_node)

        pal1_node = self.create_experiment_JSON_stub(
            subject_code=subject_code,
            experiment_name='PAL1',
            prefix='events/RAM_PAL1',
            description='Paired-Associate Learning- record-only experiment'
        )

        experiments_node.add_child_node('PAL1',pal1_node)

        pal2_node = self.create_experiment_JSON_stub(
            subject_code=subject_code,
            experiment_name='PAL2',
            prefix='events/RAM_PAL2',
            description='Paired-Associate Learning- open-loop stimulation experiment'
        )

        experiments_node.add_child_node('PAL2',pal2_node)

        pal3_node = self.create_experiment_JSON_stub(
            subject_code=subject_code,
            experiment_name='PAL3',
            prefix='events/RAM_PAL3',
            description='Paired-Associate Learning- closed-loop stimulation experiment'
        )

        experiments_node.add_child_node('PAL3',pal3_node)


        ps_node = self.create_experiment_JSON_stub(
            subject_code=subject_code,
            experiment_name='PS',
            prefix='events/RAM_PS',
            description='Parameter Search - stimulation-only task - no recal tasks'
        )
        experiments_node.add_child_node('PS',ps_node)

        # print root_node.output()
        return root_node
if __name__ == '__main__':

    # j = JSONNode()
    # j['my_task']='sdjsdhkshj'
    #
    # j1 = JSONNode()
    # j1['new_task']='djjdjdjdj'
    #
    #
    # j_tot = JSONNode()
    # j_tot.add_child_node('task1','sdsdds')
    # j_tot.add_child_node('task2',j1)
    # print j_tot.output()

    rp = RamPopulator()
    subject_dir_target = '/Users/m/data1/subjects'
    subject_list = rp.get_list_of_subjects(protocol='R1')
    # rp.create_subject_info_dir(path=subject_dir_target)
    node = rp.create_subject_JSON_stub(subject_code='R1060M')
    print node.output()


    # rp = RamPopulator()
    # subject_dir_target = '/Users/m/data1/subjects'
    # subject_list = rp.get_list_of_subjects(protocol='R1')
    # rp.create_subject_info_dir(path=subject_dir_target)
    # rp.create_subject_JSON_stub(subject_code='R1060M')
    #
    # for subject_code in subject_list:
    #     subject_node = rp.create_subject_JSON_stub(subject_code=subject_code)
    #     # print subject_node.output()
    #     subject_node.write(filename=join(subject_dir_target,subject_code,'index.json'))
    #

    # rp = RamPopulator()
    #
    # rp.mount_point = '/Users/m/data/'
    # rp.subject_dir_source = join(rp.mount_point,'eeg')
    #
    # target_dir = '/Users/m/scratch/auto_tracker'
    # subject_list = rp.get_list_of_subjects(protocol='R1')
    #
    #
    # for subject_code in subject_list:
    #     subject_node = rp.create_subject_JSON_stub(subject_code=subject_code)
    #     subject_node.write(filename=join(target_dir,subject_code,'_status','index.json'))
