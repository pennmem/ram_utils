import os
from os.path import *

from DataModel import DataLayoutJSONUtils
from JSONUtils import JSONNode
from MatlabRamTask import MatlabRamTask
from TaskRegistry import TaskRegistry


class RamPipeline(object):
    def __init__(self):
        self.task_registry = TaskRegistry()
        self.workspace_dir = ''
        self.passed_objects_dict = {}
        self.mount_point = '/'

        #  stores matlab paths
        self.matlab_paths = None

        #  flag indicating if Matlab tasks are present
        self.matlab_tasks_present = False

        self.dependency_change_tracker = None
        # self.json_saved_data_status_node = None
        # self.json_latest_status_node = None

    # def pass_object(self, name, obj):
    #     self.passed_objects_dict[name] = obj
    #
    # def get_passed_object(self, name):
    #     return self.passed_objects_dict[name]

    def set_workspace_dir(self, output_dir):
        import os

        # print 'self.output_dir=', output_dir
        output_dir_normalized = os.path.abspath(os.path.expanduser(output_dir))
        # print 'output_dir_normalized=', output_dir_normalized
        self.workspace_dir = output_dir_normalized

        try:
            os.makedirs(output_dir_normalized)
        except OSError:
            print 'Output dir: ' + output_dir_normalized + ' already exists'

    def add_task(self, task):
        task.set_pipeline(self)
        self.task_registry.register_task(task)

    def __enable_matlab(self):
        '''
        Starts Matlab Angine and sets up Matlab tasks
        :return: instance of Matlab engine
        '''
        # starts MatlabEngine
        import MatlabUtils

        # Sets up Matlab Paths
        if self.matlab_paths:
            print 'Will add the following Matlab paths: ', self.matlab_paths
            MatlabUtils.add_matlab_search_paths(*self.matlab_paths)

        # sys.exit()
        return MatlabUtils.matlab_engine

    def add_matlab_search_paths(self, paths=[]):
        '''
        Stores list of Matlab paths
        :param paths:list of matlab paths
        :return:None
        '''
        self.matlab_paths = paths

    def genrate_latest_data_status(self):
        if not self.json_saved_data_status_node:
            self.read_saved_data_status()

        if self.json_saved_data_status_node:
            subject_code = self.json_saved_data_status_node['subject']['code']
            rp = DataLayoutJSONUtils()
            rp.mount_point = self.mount_point
            self.json_latest_status_node = rp.create_subject_JSON_stub(subject_code=subject_code)
            # print self.json_latest_status_node.output()


    def get_latest_data_status(self):
        return self.json_latest_status_node

    def read_saved_data_status(self):
        json_index_file = join(self.workspace_dir,'_status','index.json')
        self.json_saved_data_status_node = JSONNode.read(filename=json_index_file)
        # rp = DataLayoutJSONUtils()
        # self.json_latest_status_node = rp.create_subject_JSON_stub(subject_code=self.)

    def get_saved_data_status(self):
        return self.json_saved_data_status_node

    def set_dependency_tracker(self,dependency_tracker):
        self.dependency_change_tracker = dependency_tracker

    def resolve_dependencies(self):


        self.dependency_change_tracker.initialize()

        if self.dependency_change_tracker:
            for task_name, task in self.task_registry.task_dict.items():

                change_flag = self.dependency_change_tracker.check_dependency_change(task)
                if change_flag:

                    try:
                        #removing task_completed_file
                        os.remove(task.get_task_completed_file_name())
                        print 'will rerun task ', task.name()
                    except OSError:
                        pass


    def prepare_matlab_tasks(self):
        for task_name, task in self.task_registry.task_dict.items():


            # task.check_dependent_resources()

            if isinstance(task, MatlabRamTask):
                if task.is_completed():
                    continue  # we do not want to start Matlab for tasks that already completed

                print 'GOT MATLAB MODULE ', task

                if not matlab_engine_started:
                    matlab_engine = self.__enable_matlab()
                    matlab_engine_started = True

                task.set_matlab_engine(matlab_engine)

    def execute_pipeline(self):
        '''
        Executes pipeline
        :return:None
        '''
        # determine if there are any of the Matlab tasks. Only then start Matlab engine
        #  checking form MatlabRamTask subclasses objects. One such objects are found we initialize matlab engine and
        #  pass matlab engine instance to those objects so that they can call matlab routines

        self.matlab_tasks_present = False
        matlab_engine_started = False
        matlab_engine = None


        self.prepare_matlab_tasks()

        if self.dependency_change_tracker:
            self.resolve_dependencies()

        # executing pipeline
        for task_name, task in self.task_registry.task_dict.items():

            # task_completed_file_name = self.get_task_completed_file_name(task)
            task_completed_file_name = task.get_task_completed_file_name()

            if task.is_completed():
                print 'RESTORING COMPLETED TASK: ', task_name, ' obj=', task
                task.restore()

            else:
                print 'RUNNING TASK: ', task_name, ' obj=', task
                task.run()
                task.copy_file_resources_to_workspace()  # copies only those resources that user requested to be copied
                task.move_file_resources_to_workspace()  # moves only those resources that user requested to be moved

                if task.mark_as_completed:
                    task.create_file_in_workspace_dir(task_completed_file_name, 'w')

        if self.dependency_change_tracker:
            self.dependency_change_tracker.write_latest_data_status()
