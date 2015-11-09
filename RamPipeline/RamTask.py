
from collections import defaultdict

class RamTask(object):

    outputs = []
    pipeline = None
    file_resources_to_copy = defaultdict()
    file_resources_to_move = defaultdict() # {file_resource:dst_dir}

    def __init__(self):
        pass

    def set_file_resources_to_copy(self,*file_resources):

        self.file_resources_to_copy = file_resources

    def set_file_resources_to_move(self,*file_resources, **kwds):
        '''
        Moves file resources to workspace dir
        :param file_resources: arguments list specifying list of files to move
        :param kwds: option supported here is dst - destination directory for the move operation
        :return:None
        '''

        for file_resource in file_resources:
            try:
                self.file_resources_to_move[file_resource] = kwds['dst']
            except LookupError:
                self.file_resources_to_move[file_resource] = ''

    def make_dir_tree(self, dirname):
        import os
        import errno
        try:
            os.makedirs(dirname)
        except OSError, e:
            if e.errno != errno.EEXIST:
                raise IOError('Could not make directory: ' + dirname)

    def copy_file_resources_to_workspace(self):

        import shutil
        import os

        for file_resource, dst_relative_path  in self.file_resources_to_copy:

            file_resource_base_name = os.path.basename(file_resource)
            shutil.copy(file_resource, os.path.join(self.pipeline.workspace_dir, dst_relative_path, file_resource_base_name))

    def move_file_resources_to_workspace(self):

        import shutil
        import os
        print 'self.file_resources_to_move=',self.file_resources_to_move
        for file_resource, dst_relative_path  in self.file_resources_to_move.items():
            # print 'file_resource=',file_resource
            if dst_path != '':
                self.make_dir_tree(os.path.join(self.pipeline.workspace_dir, dst_path))


            file_resource_base_name = os.path.basename(file_resource)
            shutil.move(file_resource, os.path.join(self.pipeline.workspace_dir, dst_relative_path, file_resource_base_name))


    def get_pipeline(self):
        return self.pipeline

    def get_workspace_dir(self):
        return self.pipeline.workspace_dir

    def set_pipeline(self,pipeline):
        self.pipeline = pipeline
    #     self.__name='Task'
    #
    # def name(self):
    #     return self.__name

    # def __set_name(self, name):
    #     self.__name = name


    def run(self):
        pass

