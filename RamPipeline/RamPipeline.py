from __future__ import print_function

import os
from os.path import *
import shutil

from DataModel import DataLayoutJSONUtils
from JSONUtils import JSONNode
from MatlabRamTask import MatlabRamTask
from TaskRegistry import TaskRegistry


class RamPipeline(object):
    def __init__(self):
        self.task_registry = TaskRegistry()
        self.workspace_dir = ''
        self.objects_dir = None  # type: str
        self.passed_objects_dict = {}
        self.mount_point = '/'

        # set to False to keep cached objects around even when the pipeline is
        # successful
        self.clear_cache_on_success = True

        # stores matlab paths
        self.matlab_paths = None

        # flag indicating if Matlab tasks are present
        self.matlab_tasks_present = False

        self.dependency_change_tracker = None

        self.exit_on_no_change = False
        self.recompute_on_no_status = False

    def set_workspace_dir(self, output_dir):
        """Sets the directory for writing data. This also creates a directory
        to cache all data that is passed to tasks further down the pipeline.

        """
        output_dir_normalized = os.path.abspath(os.path.expanduser(output_dir))
        self.workspace_dir = output_dir_normalized

        try:
            os.makedirs(output_dir_normalized)
        except OSError:
            print('Output dir: ' + output_dir_normalized + ' already exists')

        # Make directory to store objects stored with task's `pass_object` method
        self.objects_dir = join(output_dir_normalized, 'passed_objs')
        try:
            os.makedirs(join(output_dir_normalized, 'passed_objs'))
        except OSError:
            pass

    def clear_cached_objects(self):
        """Removes all cached passed objects from disk. This requires that the
        workspace directory is already set.

        """
        if self.objects_dir is None:
            raise RuntimeError("Workspace directory not yet set!")
        shutil.rmtree(self.objects_dir, True)

    def add_task(self, task):
        task.set_pipeline(self)
        self.task_registry.register_task(task)

    def __enable_matlab(self):
        """Starts Matlab engine and sets up Matlab tasks

        :return: instance of Matlab engine

        """
        # FIXME: this is bad
        # starts MatlabEngine
        import MatlabUtils

        # Sets up Matlab Paths
        if self.matlab_paths:
            print('Will add the following Matlab paths: ', self.matlab_paths)
            MatlabUtils.add_matlab_search_paths(*self.matlab_paths)

        return MatlabUtils.matlab_engine

    def add_matlab_search_paths(self, paths=[]):
        """Stores list of Matlab paths

        :param list paths: list of matlab paths
        :return: None

        """
        self.matlab_paths = paths

    # FIXME: this does not appear to be used anywhere
    def generate_latest_data_status(self):
        if not self.json_saved_data_status_node:
            self.read_saved_data_status()

        if self.json_saved_data_status_node:
            subject_code = self.json_saved_data_status_node['subject']['code']
            rp = DataLayoutJSONUtils()
            rp.mount_point = self.mount_point
            self.json_latest_status_node = rp.create_subject_JSON_stub(subject_code=subject_code)

    def get_latest_data_status(self):
        return self.json_latest_status_node

    def read_saved_data_status(self):
        json_index_file = join(self.workspace_dir, '_status', 'index.json')
        self.json_saved_data_status_node = JSONNode.read(filename=json_index_file)

    def get_saved_data_status(self):
        return self.json_saved_data_status_node

    def set_dependency_tracker(self, dependency_tracker):
        self.dependency_change_tracker = dependency_tracker

    def resolve_dependencies(self):
        self.dependency_change_tracker.initialize()

        if self.recompute_on_no_status and not self.dependency_change_tracker.is_saved_status_present():
            for task_name, task in self.task_registry.task_dict.items():
                try:
                    # removing task_completed_file
                    os.remove(task.get_task_completed_file_name())
                    print('will rerun task ', task.name())
                except OSError:
                    pass
            return 1

        new_dependency_tracking_style = False
        change_counter = 0
        for task_name, task in self.task_registry.task_dict.items():
            task_hs = task.input_hashsum()
            if task_hs != '':
                new_dependency_tracking_style = True
                completed_file_name = task.get_task_completed_file_name()
                if isfile(completed_file_name):
                    f = open(completed_file_name, 'rb')
                    hs = f.read()
                    f.close()
                    if hs != task_hs:
                        print('will rerun task ', task.name())
                        change_counter += 1

        if new_dependency_tracking_style:
            return change_counter

        change_counter = 0
        if self.dependency_change_tracker:
            for task_name, task in self.task_registry.task_dict.items():

                change_flag = self.dependency_change_tracker.check_dependency_change(task)
                if change_flag:
                    change_counter +=1
                    try:
                        # removing task_completed_file
                        os.remove(task.get_task_completed_file_name())
                        print('will rerun task ', task.name())
                    except OSError:
                        pass

        return change_counter

    def prepare_matlab_tasks(self):
        for task_name, task in self.task_registry.task_dict.items():
            if isinstance(task, MatlabRamTask):
                if task.is_completed():
                    continue  # we do not want to start Matlab for tasks that already completed

                if not matlab_engine_started:
                    matlab_engine = self.__enable_matlab()
                    matlab_engine_started = True

                task.set_matlab_engine(matlab_engine)

    def initialize_tasks(self):
        for task_name, task in self.task_registry.task_dict.items():
            task.initialize()

    def execute_pipeline(self):
        """Executes pipeline

        :return: None

        """
        # determine if there are any of the Matlab tasks. Only then start Matlab engine
        #  checking form MatlabRamTask subclasses objects. One such objects are found we initialize matlab engine and
        #  pass matlab engine instance to those objects so that they can call matlab routines

        self.matlab_tasks_present = False
        matlab_engine_started = False
        matlab_engine = None

        self.initialize_tasks()
        self.prepare_matlab_tasks()

        if self.dependency_change_tracker:
            change_counter = self.resolve_dependencies()

            if not change_counter and self.exit_on_no_change:
                return

        # executing pipeline
        for task_name, task in self.task_registry.task_dict.items():

            # task_completed_file_name = self.get_task_completed_file_name(task)
            task_completed_file_name = task.get_task_completed_file_name()

            if task.is_completed():
                print('RESTORING COMPLETED TASK: ', task_name, ' obj=', task)
                task.restore()

            else:
                print('RUNNING TASK: ', task_name, ' obj=', task)
                task.pre()
                task.run()
                task.post()
                task.copy_file_resources_to_workspace()  # copies only those resources that user requested to be copied
                task.move_file_resources_to_workspace()  # moves only those resources that user requested to be moved

                if task.mark_as_completed:
                    try:
                        hs = task.input_hashsum()
                        f = open(task_completed_file_name, 'wb')
                        f.write(hs)
                        f.close()
                    except:
                        print('No .completed file found')
                        task.create_file_in_workspace_dir(task_completed_file_name, 'w')

        if self.dependency_change_tracker:
            self.dependency_change_tracker.write_latest_data_status()

        if self.clear_cache_on_success:
            self.clear_cached_objects()
