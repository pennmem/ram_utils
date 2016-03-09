import os
from os.path import *
from JSONUtils import JSONNode

class RamPopulator(object):
    def __init__(self):
        self.mount_point = '/Users/m/data/'
        self.subject_dir_target = join(self.mount_point,'subjects')
        self.subject_dir_source = join(self.mount_point,'eeg')
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

    def create_subject_JSON_stub(self,subject):
        root_node = JSONNode()
        root_node['version'] = self.version
        root_node['subject'] = subject
        root_node['tal_bipolar']

if __name__ == '__main__':
    rp = RamPopulator()
    print rp.get_list_of_subjects(protocol='R1')
    rp.create_subject_info_dir(path='/Users/m/data1/subjects')