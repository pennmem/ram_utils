import os
import sys
from os.path import *
import re
from JSONUtils import JSONNode

from key_utils import compute_md5_key

from collections import namedtuple

SplitSubjectCode = namedtuple(typename='SplitSubjectCode',field_names=['protocol','id','site','montage'])

class RamPopulator(object):
    def __init__(self,target_root_dir='/Users/m/data1'):
        self.mount_point = '/Users/m/'
        self.subject_dir_target = join(self.mount_point,'data/subjects')
        self.subject_dir_source = join(self.mount_point,'data/eeg')
        self.protocol='R1'
        self.version = '1'

        self.target_root_dir = target_root_dir

        self.convert_subject_code_regex = re.compile('('+self.protocol+')'+'([0-9]*)([a-zA-Z]{1,1})([\S]*)')


    def get_list_of_subjects(self,protocol):
        subjects = [
            d for d in os.listdir(self.subject_dir_source)
            if isdir(join(self.subject_dir_source, d))
            and d.startswith(protocol)
                    ]
        return subjects

    # def create_subject_info_dir(self,path):
    #     subjects = self.get_list_of_subjects(protocol=self.protocol)
    #     for subject in subjects:
    #         try:
    #             os.makedirs(join(path,subject))
    #         except OSError:
    #             # in case directory exists
    #             pass




    def create_subject_info_dir(self,path):
        subjects = self.get_list_of_subjects(protocol=self.protocol)
        for subject in subjects:
            self.create_data_layout(dir=path,subject_code=subject)
            # try:
            #     os.makedirs(join(path,subject))
            # except OSError:
            #     # in case directory exists
            #     pass

    def create_symlink(self, src,target,remove_old_link='True'):


        try:
            if islink(target):
                if remove_old_link or not exists(target):
                    os.remove(target)

            if exists(src):
                os.symlink(src,target)
        except OSError:
            # in case directory exists
            print 'Could not make link between ', src, ' and ', target




    def split_subject_code(self,subject_code):
        match = re.match(self.convert_subject_code_regex,subject_code)
        if match:
            groups = match.groups()

            ssc = SplitSubjectCode(protocol=groups[0], id=groups[1],site=groups[2],montage=groups[3])
            return ssc
        return None



    def create_data_layout(self, dir, subject_code):
        # dir = self.target_root_dir

        ssc = self.split_subject_code(subject_code=subject_code)

        target_subj_dir = join(dir,'protocols',ssc.protocol.lower(),'subjects', ssc.id+ssc.montage)
        data_dir = join(target_subj_dir,'data')
        experiment_dir = join(target_subj_dir,'experiments')
        electrodes_dir = join(target_subj_dir,'electrodes')

        if not ssc:
            return

        try:
            os.makedirs(target_subj_dir)
        except OSError:
            # in case directory exists
            pass

        try:
            os.makedirs(data_dir)
        except OSError:
            # in case directory exists
            pass

        try:
            os.makedirs(experiment_dir)
        except OSError:
            # in case directory exists
            pass

        try:
            os.makedirs(electrodes_dir)
        except OSError:
            # in case directory exists
            pass


        src = join(self.mount_point, 'data', 'eeg',subject_code,'eeg.reref')
        target = join(target_subj_dir,'data','eeg.reref')
        self.create_symlink(src=src,target=target,remove_old_link='True')

        src = join(self.mount_point, 'data', 'eeg',subject_code,'eeg.noreref')
        target = join(target_subj_dir,'data','eeg.noreref')
        self.create_symlink(src=src,target=target,remove_old_link='True')


        src = join(self.mount_point, 'data', 'eeg',subject_code,'tal',subject_code+'_talLocs_database_bipol.mat')
        target = join(target_subj_dir,'electrodes','bipolar.mat')
        self.create_symlink(src=src,target=target,remove_old_link='True')

        src = join(self.mount_point, 'data', 'eeg',subject_code,'tal',subject_code+'_talLocs_database_monopol.mat')
        target = join(target_subj_dir,'electrodes','monopolar.mat')
        self.create_symlink(src=src,target=target,remove_old_link='True')

        src = join(self.mount_point, 'data', 'eeg',subject_code,'tal','good_leads.txt')
        target = join(target_subj_dir,'electrodes','good_contacts.txt')
        self.create_symlink(src=src,target=target,remove_old_link='True')

        # ----------------------- experiments ---------------------------
        self.create_experiment_data_layout(subject_code=subject_code,
                                           experiment_prefix = 'events/RAM_FR1',
                                           experiment_name = 'fr1',
                                           experiment_root_dir = experiment_dir
                                           )

        self.create_experiment_data_layout(subject_code=subject_code,
                                           experiment_prefix = 'events/RAM_FR2',
                                           experiment_name = 'fr2',
                                           experiment_root_dir = experiment_dir
                                           )


        self.create_experiment_data_layout(subject_code=subject_code,
                                           experiment_prefix = 'events/RAM_FR3',
                                           experiment_name = 'fr3',
                                           experiment_root_dir = experiment_dir
                                           )

        self.create_experiment_data_layout(subject_code=subject_code,
                                           experiment_prefix = 'events/RAM_CatFR1',
                                           experiment_name = 'catfr1',
                                           experiment_root_dir = experiment_dir
                                           )

        self.create_experiment_data_layout(subject_code=subject_code,
                                           experiment_prefix = 'events/RAM_CatFR2',
                                           experiment_name = 'catfr2',
                                           experiment_root_dir = experiment_dir
                                           )


        self.create_experiment_data_layout(subject_code=subject_code,
                                           experiment_prefix = 'events/RAM_CatFR3',
                                           experiment_name = 'catfr3',
                                           experiment_root_dir = experiment_dir
                                           )


        self.create_experiment_data_layout(subject_code=subject_code,
                                           experiment_prefix = 'events/RAM_PAL1',
                                           experiment_name = 'pal1',
                                           experiment_root_dir = experiment_dir
                                           )

        self.create_experiment_data_layout(subject_code=subject_code,
                                           experiment_prefix = 'events/RAM_PAL2',
                                           experiment_name = 'pal2',
                                           experiment_root_dir = experiment_dir
                                           )


        self.create_experiment_data_layout(subject_code=subject_code,
                                           experiment_prefix = 'events/RAM_PAL3',
                                           experiment_name = 'pal3',
                                           experiment_root_dir = experiment_dir
                                           )

        self.create_experiment_data_layout(subject_code=subject_code,
                                           experiment_prefix = 'events/RAM_PS',
                                           experiment_name = 'ps',
                                           experiment_root_dir = experiment_dir
                                           )



        # self.walk_subject_directory(target_subj_dir)

    def walk_subject_directory(self,subject_dir):

        import pathlib
        pathlib_path_parts = pathlib.Path(subject_dir).parts
        print pathlib_path_parts

        partial_path_cutoff_index=0
        for i in range(len(pathlib_path_parts)-1,0,-1):
            if pathlib_path_parts[i]=='protocols':
                partial_path_cutoff_index = i
                break


        node = JSONNode()

        for root, dirs, files in os.walk(subject_dir):

            if basename(root) == 'data':
                continue

            print 'root, dirs, files=', (root, dirs, files)
            # path = root.split('/')
            # print (len(path) - 1) *'---' , os.path.basename(root)
            print 'root=',root
            for file in files:
                root_path_parts = pathlib.Path(subject_dir).parts
                full_file_path = join(root,file)
                relative_file_path = join(root_path_parts[partial_path_cutoff_index:])
                print 'relative_file_path=',relative_file_path
                print 'file=',full_file_path

                resource_node = node
                print root_path_parts[partial_path_cutoff_index:]
                print

                for elem_name in pathlib.Path(root_path_parts[partial_path_cutoff_index:]).parts:
                    print elem_name

                    try:
                        resource_node = resource_node[elem_name]
                    except KeyError:
                        resource_node = resource_node.add_child_node(elem_name)


                resource_node['path'] = relative_file_path

                print node.output()

                print


    def create_experiment_data_layout(self,subject_code,experiment_prefix,experiment_name, experiment_root_dir):

        src = join(self.mount_point,'data',experiment_prefix,subject_code+'_events.mat')
        target_dir = join(experiment_root_dir,experiment_name.lower())

        try:
            os.makedirs(target_dir)
        except OSError:
            print 'Could not create ',target_dir


        self.create_symlink(src,join(target_dir,'events.mat'))

        src = join(self.mount_point,'data',experiment_prefix,subject_code+'_math.mat')
        target_dir = join(experiment_root_dir,experiment_name.lower())
        self.create_symlink(src,join(target_dir,'math.mat'))

        src = join(self.mount_point,'data',experiment_prefix,subject_code+'_expinfo.mat')
        target_dir = join(experiment_root_dir,experiment_name.lower())
        self.create_symlink(src,join(target_dir,'info.mat'))



    def create_experiment_JSON_stub(self, experiment_code,data_root_dir, subject_target_dir, description):
        node = JSONNode()

        p = join(subject_target_dir,'experiments',experiment_code,'events.mat')

        self.attach_single_file_JSON_stub(parent_node=node,
                                          json_stub_name='events',
                                          full_path=join(data_root_dir, p),
                                          partial_path=p
                                          )

        p = join(subject_target_dir,'experiments',experiment_code,'math.mat')
        self.attach_single_file_JSON_stub(parent_node=node,
                                          json_stub_name='math',
                                          full_path=join(data_root_dir, p),
                                          partial_path=p
                                          )


        p = join(subject_target_dir,'experiments',experiment_code,'info.mat')
        self.attach_single_file_JSON_stub(parent_node=node,
                                          json_stub_name='info',
                                          full_path=join(data_root_dir, p),
                                          partial_path=p
                                          )


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

        ssc = self.split_subject_code(subject_code=subject_code)

        if not ssc:
            return


        subject_target_dir = join('protocols',ssc.protocol.lower(),'subjects',str(ssc.id))

        data_root_dir = join(self.mount_point,'data1')

        root_node = JSONNode()
        root_node['version'] = self.version

        subject_node = root_node.add_child_node('subject')
        subject_node['protocol'] = ssc.protocol.lower()
        subject_node['id'] = ssc.id
        subject_node['site'] = ssc.site
        subject_node['montage'] = ssc.montage

        # ------------------- electrodes ---------------------------------
        electrodes_info = root_node.add_child_node('electrodes')

        p = join('eeg',subject_code,'tal',subject_code+'_talLocs_database_bipol.mat')



        p = join(subject_target_dir,'electrodes','bipolar.mat')
        self.attach_single_file_JSON_stub(parent_node=electrodes_info,
                                          json_stub_name='bipolar',
                                          full_path=join(data_root_dir,p),
                                          partial_path=p)


        p = join(subject_target_dir,'electrodes','monopolar.mat')
        self.attach_single_file_JSON_stub(parent_node=electrodes_info,
                                          json_stub_name='monopolar',
                                          full_path=join(data_root_dir,p),
                                          partial_path=p)

        # --------------------- eeg ---------------------------------
        eeg_node = root_node.add_child_node('eeg')
        eeg_reref_node = eeg_node.add_child_node('reref_dir')
        eeg_reref_node['path'] = join('eeg',subject_code,'eeg.reref')

        eeg_noreref_node = eeg_node.add_child_node('noreref_dir')
        eeg_noreref_node['path'] = join('eeg',subject_code,'eeg.noreref')


        p = join('eeg',subject_code,'eeg.reref','params.txt')
        p = join(subject_target_dir,'data','eeg.reref','params.txt')
        self.attach_single_file_JSON_stub(parent_node=eeg_node,
                                          json_stub_name='params_reref',
                                          full_path=join(data_root_dir,p),
                                          partial_path=p)

        p = join(subject_target_dir,'data','eeg.noreref','params.txt')
        self.attach_single_file_JSON_stub(parent_node=eeg_node,
                                          json_stub_name='params_noreref',
                                          full_path=join(data_root_dir,p),
                                          partial_path=p)



        # --------------------- experiments ---------------------------------
        experiments_node = root_node.add_child_node('experiments')



        fr1_node = self.create_experiment_JSON_stub(
            experiment_code='fr1',
            data_root_dir=data_root_dir,
            subject_target_dir=subject_target_dir,
            description='Free Recall - record-only experiment'
        )

        experiments_node.add_child_node('fr1',fr1_node)



        fr2_node = self.create_experiment_JSON_stub(
            experiment_code='fr2',
            data_root_dir=data_root_dir,
            subject_target_dir=subject_target_dir,
            description='Free Recall - open-loop stimulation  experiment'
        )

        experiments_node.add_child_node('fr2',fr2_node)


        fr3_node = self.create_experiment_JSON_stub(
            experiment_code='fr3',
            data_root_dir=data_root_dir,
            subject_target_dir=subject_target_dir,
            description='Free Recall - closed-loop stimulation  experiment'
        )

        experiments_node.add_child_node('fr3',fr3_node)



        catfr1_node = self.create_experiment_JSON_stub(
            experiment_code='catfr1',
            data_root_dir=data_root_dir,
            subject_target_dir=subject_target_dir,
            description='Free Recall - record-only experiment'
        )

        experiments_node.add_child_node('catfr1',catfr1_node)

        catfr2_node = self.create_experiment_JSON_stub(
            experiment_code='catfr2',
            data_root_dir=data_root_dir,
            subject_target_dir=subject_target_dir,
            description='Free Recall - open-loop stimulation  experiment'
        )

        experiments_node.add_child_node('catfr2',catfr2_node)


        catfr3_node = self.create_experiment_JSON_stub(
            experiment_code='catfr3',
            data_root_dir=data_root_dir,
            subject_target_dir=subject_target_dir,
            description='Free Recall - closed-loop stimulation  experiment'
        )

        experiments_node.add_child_node('catfr3',catfr3_node)


        pal1_node = self.create_experiment_JSON_stub(
            experiment_code='pal1',
            data_root_dir=data_root_dir,
            subject_target_dir=subject_target_dir,
            description='Paired-Associates Learning - record-only experiment'
        )

        experiments_node.add_child_node('pal1',pal1_node)

        pal2_node = self.create_experiment_JSON_stub(
            experiment_code='pal2',
            data_root_dir=data_root_dir,
            subject_target_dir=subject_target_dir,
            description='Paired-Associates Learning - open-loop stimulation  experiment'
        )

        experiments_node.add_child_node('pal2',pal2_node)


        pal3_node = self.create_experiment_JSON_stub(
            experiment_code='pal3',
            data_root_dir=data_root_dir,
            subject_target_dir=subject_target_dir,
            description='Paired-Associates Learning - closed-loop stimulation  experiment'
        )

        experiments_node.add_child_node('pal3',pal3_node)

        ps_node = self.create_experiment_JSON_stub(
            experiment_code='ps',
            data_root_dir=data_root_dir,
            subject_target_dir=subject_target_dir,
            description='Parameter Search - stimulation-only task - no recal tasks'
        )
        experiments_node.add_child_node('ps',ps_node)



        print root_node.output()
        return root_node



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

    # rp = RamPopulator()
    # rp.split_subject_code('r1060M_1')
    # sys.exit()

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

    # rp = RamPopulator()
    # subject_dir_target = '/Users/m/data1/subjects'
    # subject_list = rp.get_list_of_subjects(protocol='R1')
    # # rp.create_subject_info_dir(path=subject_dir_target)
    # node = rp.create_subject_JSON_stub(subject_code='R1060M')
    # print node.output()


    rp = RamPopulator()
    subject_dir_target = '/Users/m/data1/'
    subject_list = rp.get_list_of_subjects(protocol='R1')
    rp.create_subject_info_dir(path=subject_dir_target)
    rp.create_subject_JSON_stub(subject_code='R1060M')

    for subject_code in subject_list:
        subject_node = rp.create_subject_JSON_stub(subject_code=subject_code)
        # print subject_node.output()
        subject_node.write(filename=join(subject_dir_target,'protocols',
                                         subject_node['subject']['protocol'],
                                         'subjects',
                                         subject_node['subject']['id']+subject_node['subject']['montage'],
                                        'index.json')
                            )


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







