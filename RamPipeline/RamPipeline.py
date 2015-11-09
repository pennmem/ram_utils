
from TaskRegistry import TaskRegistry

class RamPipeline(object):

    def __init__(self):
        self.task_registry = TaskRegistry()
        self.workspace_dir = ''




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


    def open_file_in_workspace_dir(self,file_name, mode='r'):
        from os.path import abspath, join

        try:
            open(abspath(join(self.workspace_dir, file_name)), mode)
        except IOError:
            return None, None


    def create_file_in_workspace_dir(self, file_name, mode='w'):
        import os
        from distutils.dir_util import mkpath


        output_file_name = os.path.join(self.workspace_dir, file_name)
        output_file_name = os.path.abspath(output_file_name)# normalizing path
        dir_for_output_file_name = os.path.dirname(output_file_name)

        try:
            mkpath(dir_for_output_file_name )
        except:
            raise IOError

        try :
            return open(output_file_name, mode),output_file_name
        except:
            raise IOError ('COULD NOT OPEN '+output_file_name+' in mode='+mode)



    def create_dir_in_workspace(self, dir_name):

        import os

        if self.workspace_dir == '':

            raise RuntimeError('You must set up root of output directory first before creating additional directories.\
             Use set_output_dir_root function')


        try:
            dir_name_normalized = os.path.abspath(os.path.join(self.workspace_dir, dir_name))

            os.makedirs(dir_name_normalized)
        except OSError:
            print 'skipping: '+dir_name_normalized+ ' perhaps it already exists'
            pass

        return dir_name_normalized

    def create_multiple_dirs_in_workspace(self, *dir_names):
        import os

        if self.workspace_dir == '':

            raise RuntimeError('You must set up root of output directory first before creating additional directories.\
             Use set_output_dir_root function')

        for dir_name in dir_names:
            print dir_name
            try:
                dir_name_normalized = os.path.abspath(os.path.join(self.workspace_dir, dir_name))

                os.makedirs(dir_name_normalized)
            except OSError:
                print 'skipping: '+dir_name_normalized+ ' perhaps it already exists'
                pass

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
                self.create_file_in_workspace_dir(task_completed_file_name,'w')


        # self.task_registry.run_tasks()
