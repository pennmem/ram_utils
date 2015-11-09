
from collections import defaultdict

class RamTask(object):

    outputs = []
    pipeline = None
    file_resources_to_copy = defaultdict()
    file_resources_to_move = defaultdict() # {file_resource:dst_dir}

    def __init__(self):
        pass

    def set_file_resources_to_copy(self, *file_resources, **kwds):
        '''
        setter method that allows users to specify file resources and subfolder of the workspace dir
        where the file respirces should be copied
        :param file_resources: arguments list specifying list of files to copy
        :param kwds: dst - destination directory for the move operation - the option supported here
        :return:None
        '''

        for file_resource in file_resources:
            try:
                self.file_resources_to_copy[file_resource] = kwds['dst']
            except LookupError:
                self.file_resources_to_copy[file_resource] = ''

    def set_file_resources_to_move(self, *file_resources, **kwds):
        '''
        setter method that allows users to specify file resources and subfolder of the workspace dir
        where the file respirces should be moved
        :param file_resources: arguments list specifying list of files to move
        :param kwds: dst - destination directory for the move operation - the option supported here
        :return:None
        '''

        for file_resource in file_resources:
            try:
                self.file_resources_to_move[file_resource] = kwds['dst']
            except LookupError:
                self.file_resources_to_move[file_resource] = ''

    def make_dir_tree(self, dirname):
        '''
        attempts to create directory tree
        :param dirname: directory name
        :return:None
        '''
        import os
        import errno
        try:
            os.makedirs(dirname)
        except OSError, e:
            if e.errno != errno.EEXIST:
                raise IOError('Could not make directory: ' + dirname)

    def copy_file_resources_to_workspace(self):
        '''
        Examines dictionary of files to copy and copies listed files to the appropriate workspace folder or its subfolder
        :return:None
        '''

        import shutil
        import os

        for file_resource, dst_relative_path in self.file_resources_to_copy:
            if dst_relative_path != '':
                self.make_dir_tree(os.path.join(self.pipeline.workspace_dir, dst_relative_path))

            file_resource_base_name = os.path.basename(file_resource)
            shutil.copy(file_resource, os.path.join(self.pipeline.workspace_dir, dst_relative_path, file_resource_base_name))

    def move_file_resources_to_workspace(self):
        '''
        Examines dictionary of files to move and moves listed files to the appropriate workspace folder or its subfolder
        :return:None
        '''

        import shutil
        import os

        for file_resource, dst_relative_path in self.file_resources_to_move.items():

            if dst_relative_path != '':
                self.make_dir_tree(os.path.join(self.pipeline.workspace_dir, dst_relative_path))

            file_resource_base_name = os.path.basename(file_resource)
            shutil.move(file_resource, os.path.join(self.pipeline.workspace_dir, dst_relative_path, file_resource_base_name))


    def get_pipeline(self):
        '''
        Returns pipeline object to which a given task belongs to
        :return:pipeline object
        '''
        return self.pipeline

    def get_workspace_dir(self):
        '''
        Returns full path to the workspace dir
        :return: full path to the workspace dir
        '''
        return self.pipeline.workspace_dir

    def set_pipeline(self,pipeline):
        '''
        Initializes reference pointing to the pipeline object to which current task belongs to.
        This initialization is done automatically when user adds task to the pipeline
        :param pipeline: pipeline
        :return:None
        '''

        self.pipeline = pipeline
    #     self.__name='Task'
    #
    # def name(self):
    #     return self.__name

    # def __set_name(self, name):
    #     self.__name = name


    def run(self):
        '''
        Core function of the task object - needs to be reimplmented in each subslacc deriving from RamTask
        :return:None
        '''
        pass

