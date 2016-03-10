from DataMonitor import RamPopulator
from JSONUtils import JSONNode
from DataMonitor import compute_md5_key
from os.path import *

class DependencyChangeTracker(object):
    def __init__(self):
        self.json_saved_data_status_node = None
        self.json_latest_status_node = None
        self.mount_point = ''
        self.workspace_dir = ''

    def initialize(self):
        self.genrate_latest_data_status()

    def genrate_latest_data_status(self):
        if not self.json_saved_data_status_node:
            self.read_saved_data_status()

        if self.json_saved_data_status_node:
            subject_code = self.json_saved_data_status_node['subject']['code']
            rp = RamPopulator()
            rp.mount_point = self.mount_point
            self.json_latest_status_node = rp.create_subject_JSON_stub(subject_code=subject_code)
            # print self.json_latest_status_node.output()


    def get_latest_data_status(self):
        return self.json_latest_status_node

    def read_saved_data_status(self):
        json_index_file = join(self.workspace_dir,'_status','index.json')
        self.json_saved_data_status_node = JSONNode.read(filename=json_index_file)
        # rp = RamPopulator()
        # self.json_latest_status_node = rp.create_subject_JSON_stub(subject_code=self.)

    def get_saved_data_status(self):
        return self.json_saved_data_status_node

    def write_latest_data_status(self):
        if self.json_latest_status_node:
            self.json_latest_status_node.write(join(self.workspace_dir,'_status','index.json'))

    def dependency_change(self, task):
        change_flag = False

        dependency_inventory = task.get_dependency_inventory()
        if not dependency_inventory:
            return change_flag

        status_json_node = self.get_saved_data_status()
        # status_json_node = self.read_status()


        if status_json_node is None:
            print 'Could not find saved file status json stub'
            return change_flag

        for resource_name, json_node_access_list in dependency_inventory:
            resource_node = status_json_node
            for node_name in json_node_access_list:
                try:
                    resource_node = resource_node[node_name]
                except KeyError:
                    raise KeyError('Could not locate node = ' + node_name)

            full_resource_path = join(self.mount_point,'data', resource_node['path'])
            print 'full_resource_path=',full_resource_path
            md5 = compute_md5_key(full_resource_path)

            if md5 != resource_node['md5']:
                print 'Dependency for task =',task.name(), ' has changed'
                change_flag = True
                return change_flag
                # os.remove(self.get_task_completed_file_name())
            # print resource_node.output()


