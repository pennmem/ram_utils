
from TaskRegistry import TaskRegistry
from os.path import *


class RamPipeline(object):

    def __init__(self):
        self.task_registry = TaskRegistry()
        self.workspace_dir = ''
        self.passed_objects_dict={}

    def add_object_to_pass(self, name, obj):
        self.passed_objects_dict[name] = obj

    def get_passed_object(self,name):
        return self.passed_objects_dict[name]

    def set_workspace_dir(self, output_dir):
        import os

        print 'self.output_dir=',output_dir
        output_dir_normalized = os.path.abspath(os.path.expanduser(output_dir))
        print 'output_dir_normalized=',output_dir_normalized
        self.workspace_dir = output_dir_normalized

        try:
            os.makedirs(output_dir_normalized)
        except OSError:
            print 'Output dir: '+output_dir_normalized+' already exists'

    def add_task(self,task):
        task.set_pipeline(self)
        self.task_registry.register_task(task)

    def execute_pipeline(self):
        import os

        for task_name, task in self.task_registry.task_dict.items():

            task_completed_file_name = os.path.join(self.workspace_dir,task_name+'.completed')

            if os.path.isfile( task_completed_file_name):
                print 'SKIPPING COMPLETED TASK: ', task_name,' obj=',task

            else:
                print 'RUNNING TASK: ', task_name,' obj=',task
                task.run()
                task.copy_file_resources_to_workspace() # copies only those resources that user requested to be copied
                task.move_file_resources_to_workspace() # moves only those resources that user requested to be moved

                if task.mark_as_completed:
                    task.create_file_in_workspace_dir(task_completed_file_name,'w')
                    # self.create_file_in_workspace_dir(task_completed_file_name,'w')


        # self.task_registry.run_tasks()
