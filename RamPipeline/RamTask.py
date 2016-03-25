from collections import OrderedDict
from collections import defaultdict
import os
from os.path import *

from distutils.dir_util import mkpath

from JSONUtils import JSONNode
from DataModel import compute_md5_key
from DependencyInventory import DependencyInventory


class RamTask(object):
    def __init__(self, mark_as_completed=True):
        self.outputs = []
        self.pipeline = None
        self.workspace_dir = None
        self.file_resources_to_copy = defaultdict()
        self.file_resources_to_move = defaultdict()  # {file_resource:dst_dir}
        self.mark_as_completed = True
        self.__name = None

        self.set_mark_as_completed(mark_as_completed)

        self.dependency_inventory = DependencyInventory()

        # self.__dependent_resources__ = OrderedDict()

    def get_dependency_inventory(self):
        return self.dependency_inventory

    # def read_status(self):
    #     json_index_file = join(self.workspace_dir,'_status','index.json')
    #     return JSONNode.read(filename=json_index_file)

    # def add_dependent_resource(self,resource_name,json_node_access_list):
    #     self.__dependent_resources__[resource_name] = json_node_access_list

    # def check_dependent_resources(self):
    #
    #     status_json_node = self.pipeline.get_saved_data_status()
    #     # status_json_node = self.read_status()
    #     if status_json_node is None:
    #         print 'Could not findsaved file status json stub'
    #         return
    #     for resource_name, json_node_access_list in self.__dependent_resources__.items():
    #         resource_node = status_json_node
    #         for node_name in json_node_access_list:
    #             try:
    #                 resource_node = resource_node[node_name]
    #             except KeyError:
    #                 raise KeyError('Could not locate node = ' + node_name)
    #
    #         full_resource_path = join(self.pipeline.mount_point,'data', resource_node['path'])
    #         print 'full_resource_path=',full_resource_path
    #         md5 = compute_md5_key(full_resource_path)
    #
    #         if md5 != resource_node['md5']:
    #             print 'RESOUCE HAS CHANGED .RERUNNING TASK!!!!!!!!!!!!!!!!!!'
    #             os.remove(self.get_task_completed_file_name())
    #         print resource_node.output()



    def set_name(self, name):
        self.__name = name

    def name(self):
        return self.__name

    def pass_object(self, name, obj):
        self.pipeline.passed_objects_dict[name] = obj

    def get_passed_object(self, name):
        return self.pipeline.passed_objects_dict[name]

    def get_task_completed_file_name(self):
        '''
        retunrs name of the task
        :param task: task object object derived from RamTask or MatlabRamTask
        :return: task name - this is the name of the derived class
        '''

        return join(self.workspace_dir, self.name() + '.completed')

    def check_json_stub(self):
        json_stub_file = join(self.workspace_dir, 'index.json')

        print 'json stub=', JSONNode.read(filename=json_stub_file)
        print

    # def read_status(self):
    #     json_index_file = join(self.workspace_dir,'_status','index.json')
    #     self.json_index_node = JSONNode.read(filename=json_index_file)
    #
    # def get_json_status_node(self):
    #     return self.json_index_node



    def is_completed(self):
        '''
        returns flag indicating if the task was completed or not
        :param task: task object - object derived from RamTask or MatlabRamTask
        :return: bool indicating if the file marking the completeion of hte task is present or not.
        In the Future we will use SHA1 key for more robust completion of the completed task
        '''
        # self.check_json_stub()
        return isfile(self.get_task_completed_file_name())

    def set_pipeline(self, pipeline):
        """
        Sets reference to the pipeline
        :param pipeline: pipeline object
        :return:None
        """
        self.pipeline = pipeline
        self.workspace_dir = self.set_workspace_dir(self.pipeline.workspace_dir)

    def set_workspace_dir(self, workspace_dir):
        """
        Sets Workspace dir
        :param workspace_dir: full path to the workspace dir
        :return:None
        """
        self.workspace_dir = workspace_dir

    def set_mark_as_completed(self, flag):
        '''
        This function is used to inform the executing pipeline whether to mark task as completed in the workspace dir or not
        Calling set_mark_as_completed(False) will disable taging task as completed
        so that each time pipeline is run this task will be invoked. This is very useful during debuggin stage where
        usually one would disable marking of the task as completed and only when the code is ready to run in te "production mode"
        the call to set_mark_as_completed(False) woudl get removed so that marking of the task completion is reinstated
        :param flag:boolean flag
        :return:None
        '''
        self.mark_as_completed = flag

    def open_file_in_workspace_dir(self, file_name, mode='r'):
        """
        Opens file in the workspace directory - the default file open mode is 'r'
        :param file_name: file name relative to the workspace directory
        :param mode: file open mode
        :return: (file object, full_path_to the file)
        """
        assert self.workspace_dir is not None, "Workspace directory was not set"

        try:

            full_file_name = abspath(join(self.workspace_dir, file_name))
            file = open(full_file_name, mode)
            return file, full_file_name

        except IOError:
            return None, None

    # def create_file_in_workspace_dir(self, file_name, mode='w'):
    #     """
    #     Creates file in the workspace directory - the default file open mode is 'w'.
    #     In case certain elements of the path are missing they will be created.
    #     :param file_name: file name relative to the workspace directory
    #     :param mode: file open mode
    #     :return: (file object, full_path_to the file)
    #     """
    #     assert self.workspace_dir is not None, "Workspace directory was not set"
    #
    #     output_file_name = os.path.join(self.workspace_dir, file_name)
    #     output_file_name = os.path.abspath(output_file_name)# normalizing path
    #     dir_for_output_file_name = os.path.dirname(output_file_name)
    #
    #     try:
    #         mkpath(dir_for_output_file_name)
    #     except:
    #         raise IOError('Could not create directory path %s'%dir_for_output_file_name)
    #
    #     try:
    #         return open(output_file_name, mode),output_file_name
    #     except:
    #         raise IOError ('COULD NOT OPEN '+output_file_name+' in mode='+mode)

    def create_file_in_workspace_dir(self, file_name, mode='w'):
        """
        Creates file in the workspace directory - the default file open mode is 'w'.
        In case certain elements of the path are missing they will be created.
        :param file_name: file name relative to the workspace directory
        :param mode: file open mode
        :return: (file object, full_path_to the file)
        """

        file_name_to_file_obj_full_path_dict = self.create_multiple_files_in_workspace_dir(file_name, mode=mode)

        try:
            return file_name_to_file_obj_full_path_dict[file_name]  # returns a tuple (file object, full file name)
        except LookupError:
            raise LookupError('Could not locate file_name: %s  in the dictionary of created files' % file_name)

    def create_multiple_files_in_workspace_dir(self, *rel_file_names, **options):
        """
        Creates multiple file names in the workspace
        :param rel_file_names: comma-separated list of file names relative to the workspacedir
        :param options:
        default option is mode = 'w'. Other options can be specified using mode='file mode'
        :return: dictionary {relative_file_path:(file object, full_path_to_created_file)}
        """

        assert self.workspace_dir is not None, "Workspace directory was not set"

        try:
            mode = options['mode']
        except LookupError:
            mode = 'w'

        file_name_to_file_obj_full_path_dict = {}

        for rel_file_name in rel_file_names:

            output_file_name = join(self.workspace_dir, rel_file_name)
            output_file_name = abspath(output_file_name)  # normalizing path
            dir_for_output_file_name = dirname(output_file_name)

            try:
                mkpath(dir_for_output_file_name)
            except:
                raise IOError('Could not create directory path %s' % dir_for_output_file_name)

            try:
                file_name_to_file_obj_full_path_dict[rel_file_name] = (open(output_file_name, mode), output_file_name)
                # return open(output_file_name, mode),output_file_name
            except IOError:
                raise IOError('COULD NOT OPEN ' + output_file_name + ' in mode=' + mode)

        return file_name_to_file_obj_full_path_dict

    def create_dir_in_workspace(self, dir_name):
        """
        Creates directory in the workspace using
        :param dir_name: directory path relative to the workspace_dir
        :return: full path to the created directory
        """

        dir_name_dict = self.create_multiple_dirs_in_workspace(dir_name)
        try:
            return dir_name_dict[dir_name]
        except LookupError:
            return None

    def create_multiple_dirs_in_workspace(self, *dir_names):
        """
        Creates multiple directories in the workspace
        :param dir_names: comma separated list of the directory paths relative to the workspace_dir
        :return: dictionary {relative_dir_name:full_path_to_created_dir}
        """

        assert self.workspace_dir is not None, "Workspace directory was not set"
        dir_name_dict = {}
        for dir_name in dir_names:
            # print dir_name
            try:
                dir_name_full_path = abspath(join(self.workspace_dir, dir_name))
                os.makedirs(dir_name_full_path)
                dir_name_dict[dir_name] = dir_name_full_path

            except OSError:
                print 'skipping: ' + dir_name_full_path + ' perhaps it already exists'
                pass

        return dir_name_dict

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

        for file_resource, dst_relative_path in self.file_resources_to_copy.items():
            if dst_relative_path != '':
                self.make_dir_tree(os.path.join(self.pipeline.workspace_dir, dst_relative_path))

            file_resource_base_name = os.path.basename(file_resource)
            try:
                target_path = os.path.abspath(
                    os.path.join(self.pipeline.workspace_dir, dst_relative_path, file_resource_base_name))
                shutil.copy(file_resource, target_path)
            except IOError:
                print 'Could not copy file: ', file_resource, ' to ', target_path

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
            try:
                target_path = os.path.abspath(
                    os.path.join(self.pipeline.workspace_dir, dst_relative_path, file_resource_base_name))
                shutil.move(file_resource, target_path)
            except IOError:
                print 'Could not move file: ', file_resource, ' to ', target_path

    def get_path_to_resource_in_workspace(self, *rel_path_components):
        """
        Returns absolute path to the rel_path_component assuming that rel_path_component is specified w.r.t workspace_dir
        :param rel_path_components: path relative to the workspace dir
        :return:absolute path
        """

        assert self.workspace_dir is not None, "Workspace directory was not set"

        return abspath(join(self.workspace_dir, *rel_path_components))

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
        return self.workspace_dir

    def set_pipeline(self, pipeline):
        '''
        Initializes reference pointing to the pipeline object to which current task belongs to.
        This initialization is done automatically when user adds task to the pipeline
        :param pipeline: pipeline
        :return:None
        '''

        self.pipeline = pipeline
        try:
            self.set_workspace_dir(self.pipeline.workspace_dir)
        except AttributeError:
            raise AttributeError('The pipeline you are trying to set has no workspace_dir attribute')

        assert self.get_workspace_dir() is not None, 'After setting the pipeline the workspace is still None.' \
                                                     ' Perhaps you need to specify workspace_dir in the pipeline object'

    #     self.__name='Task'
    #
    # def name(self):
    #     return self.__name

    # def __set_name(self, name):
    #     self.__name = name

    def initialize(self):
        pass

    def restore(self):
        '''
        Core function that restores saved results of the task so that following tasks in the pipeline can continue
        :return:
        '''
        pass

    def pre(self):
        """
        Core function will be called before run - can be used in subclasses to run code just before run object gets called
        :return:None
        """
        pass

    def post(self):
        """
        Core function will be called before run - can be used in subclasses to run code right after run object gets called
        :return:None
        """
        pass

    def run(self):
        '''
        Core function of the task object - needs to be reimplmented in each subslacc deriving from RamTask
        :return:None
        '''
        pass


if __name__ == '__main__':
    rt = RamTask()
    rt.set_workspace_dir('/Users/m/my_workspace')
    print 'rt.workspace_dir = ', rt.workspace_dir
    print 'rt.get_workspace_dir=', rt.get_workspace_dir()

    print 'get_path_to_file_in_workspace = ', rt.get_path_to_resource_in_workspace('abc/cba/cbos', 'mst')

    # print 'this is get_path_to_file_in_workspace=',rt.get_path_to_file_in_workspace('demo1')
