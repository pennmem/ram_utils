from DataModel import DataLayoutJSONUtilsLegacy
from JSONUtils import JSONNode
from DataModel import compute_md5_key
from os.path import *

from RamPipeline.DependencyChangeTrackerBase import DependencyChangeTrackerBase
import warnings
from RamPipeline import RamResource
from collections import OrderedDict


class DependencyChangeTrackerLegacy(DependencyChangeTrackerBase):

    def __init__(self, *args, **kwds):
        super(DependencyChangeTrackerLegacy,self).__init__(*args, **kwds)
        try:
            self.subject = kwds['subject']
        except KeyError:
            self.subject = ''

        try:
            self.workspace_dir = kwds['workspace_dir']
        except KeyError:
            self.workspace_dir = ''

        try:
            self.mount_point = kwds['mount_point']
        except KeyError:
            self.mount_point = ''

        self.json_saved_data_status_node = None
        self.json_latest_status_node = None

    def initialize(self):
        self.generate_latest_data_status()

    def generate_latest_data_status(self):
        if not self.json_saved_data_status_node:
            self.read_saved_data_status()

        # if self.json_saved_data_status_node:

        # subject_code = self.json_saved_data_status_node['subject']['code']
        subject_code = self.subject
        rp = DataLayoutJSONUtilsLegacy()
        rp.mount_point = self.mount_point
        self.json_latest_status_node = rp.create_subject_JSON_stub(subject_code=subject_code)
        # print self.json_latest_status_node.output()


        if not self.json_saved_data_status_node:  # if saved json stub was not found
            self.json_saved_data_status_node = self.json_latest_status_node

    def get_latest_data_status(self):
        return self.json_latest_status_node

    def is_saved_status_present(self):
        if isfile(join(self.workspace_dir, '_status', 'index.json')):
            return True
        else:
            return False

    def read_saved_data_status(self):
        json_index_file = join(self.workspace_dir, '_status', 'index.json')
        self.json_saved_data_status_node = JSONNode.read(filename=json_index_file)
        # rp = DataLayoutJSONUtils()
        # self.json_latest_status_node = rp.create_subject_JSON_stub(subject_code=self.)

    def get_saved_data_status(self):
        return self.json_saved_data_status_node

    def write_latest_data_status(self):
        if self.json_latest_status_node:
            self.json_latest_status_node.write(join(self.workspace_dir, '_status', 'index.json'))

    def check_dependency_change(self, task):
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
                    warnings.warn("Could not locate node " + node_name, RuntimeWarning)
                    self.changed_resources[' -> '.join(json_node_access_list)] = RamResource(name=' -> '.join(json_node_access_list),task=task.name(),status='non_existent_resource')


                    change_flag = True
                    return change_flag

                    # raise KeyError('Could not locate node = ' + node_name)

            full_resource_path = join(self.mount_point, 'data', resource_node['path'])
            print 'full_resource_path=', full_resource_path
            md5 = compute_md5_key(full_resource_path)

            if md5 != resource_node['md5']:
                print 'Dependency for task =', task.name(), ' has changed'
                self.changed_resources[full_resource_path] = RamResource(name=full_resource_path,task=task.name(),status='changed_existing_resource')

                change_flag = True
                return change_flag
                # os.remove(self.get_task_completed_file_name())
                # print resource_node.output()
