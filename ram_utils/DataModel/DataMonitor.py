import os
from os.path import *
import collections
import hashlib
from JSONUtils.JSONNode import *

class DataMonitor(object):

    def __init__(self):
        self.root = '/Users/m/ROOT'
        self.subject_dir = join(self.root,'protocols/r1/subjects')

    def get_subjects_info(self):
        dir_list = next(os.walk(self.subject_dir))[1]

        subject_dict = collections.OrderedDict()
        for dir in dir_list:
            index_path = abspath(join(self.subject_dir,dir,'index.json'))

            if isfile(index_path):
                subject_dict[dir] = index_path

            # if isdir(join(self.subject_dir,dir,'data')):
            #     print join(self.subject_dir,dir,'data')
            #     # join(self.subject_dir,dir
        # print dir_list
        #
        # print subject_dict
        return subject_dict


    def compute_md5_key(self,filename):
        hash_md5 = hashlib.md5()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def generate_json_stub(self):
        resource_name_list = [('tal_bipol','tal_bipol.mat')]

        subject_dict = self.get_subjects_info()
        for subject_code, index_filename in subject_dict.items():
            subject_info_dir = dirname(index_filename)

            subject_jnode = JSONNode()


            for (resource_name,resource_filename) in resource_name_list:
                resource_path = join(subject_info_dir,resource_filename)
                print resource_path
                if isfile(resource_path):

                    print (resource_name,resource_filename)
                    resource_jnode = JSONNode()
                    resource_jnode['path'] = resource_path
                    resource_jnode['md5'] = self.compute_md5_key(resource_path)

                    subject_jnode[resource_name]=resource_jnode

            print subject_jnode.write(join(subject_info_dir,'idx.json'))

    def check_changes(self):
        pass


dm = DataMonitor()

dm.get_subjects_info()

dm.generate_json_stub()