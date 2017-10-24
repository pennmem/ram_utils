from __future__ import print_function

import os
import os.path as osp
import shutil
from collections import defaultdict, OrderedDict
from distutils.dir_util import mkpath

from sklearn.externals import joblib


class RamTask(object):
    def __init__(self, mark_as_completed=True, force_rerun=False):
        """
        :param bool mark_as_completed:
            Mark the task as completed upon completion.
        :param bool force_rerun:
            Force a rerun even if the task was already marked as complete.

        """
        self.outputs = []
        self.pipeline = None
        self.workspace_dir = None
        self.file_resources_to_copy = defaultdict()
        self.file_resources_to_move = defaultdict()  # {file_resource:dst_dir}
        self.mark_as_completed = mark_as_completed
        self.__force_rerun = force_rerun
        self.__name = None

        self.set_mark_as_completed(mark_as_completed)

    def input_hashsum(self):
        return ''

    def set_name(self, name):
        self.__name = name

    def name(self):
        return self.__name

    def _obj_filename(self, name):
        return osp.join(self.pipeline.objects_dir, "{}.pkl".format(name))

    def pass_object(self, name, obj):
        """Makes an object accessible further down the pipeline."""
        # Store on disk
        joblib.dump(obj, self._obj_filename(name))

        # Store in memory
        self.pipeline.passed_objects_dict[name] = obj

    def get_passed_object(self, name):
        """Loads a passed object from earlier in the pipeline."""
        try:
            # Try loading from memory
            obj = self.pipeline.passed_objects_dict[name]
        except KeyError:
            try:
                obj = joblib.load(self._obj_filename(name))
            except:
                raise RuntimeError("Could not find passed object {} in memory or on disk".format(name))
        return obj

    def get_task_completed_file_name(self):
        """
        retunrs name of the task
        :param task: task object object derived from RamTask or MatlabRamTask
        :return: task name - this is the name of the derived class
        """
        return osp.join(self.workspace_dir, self.name() + '.completed')

    def check_json_stub(self):
        json_stub_file = osp.join(self.workspace_dir, 'index.json')
        print('json stub=', JSONNode.read(filename=json_stub_file))

    def is_completed(self):
        """
        returns flag indicating if the task was completed or not
        :param task: task object - object derived from RamTask or MatlabRamTask
        :return: bool indicating if the file marking the completeion of the task is present
        and if the dependency hashsum stored in it is equal to the current dependency hashsum
        """
        if self.__force_rerun:
            return False

        completed_file_name = self.get_task_completed_file_name()
        if osp.isfile(completed_file_name):
            f = open(completed_file_name, 'rb')
            hs = f.read()
            f.close()
            return hs == self.input_hashsum()
        else:
            return False

    def set_pipeline(self, pipeline):
        """
        Initializes reference pointing to the pipeline object to which current task belongs to.
        This initialization is done automatically when user adds task to the pipeline
        :param pipeline: pipeline
        :return:None
        """
        self.pipeline = pipeline
        try:
            self.set_workspace_dir(self.pipeline.workspace_dir)
        except AttributeError:
            raise AttributeError('The pipeline you are trying to set has no workspace_dir attribute')

        assert self.get_workspace_dir() is not None, 'After setting the pipeline the workspace is still None.' \
                                                     ' Perhaps you need to specify workspace_dir in the pipeline object'

    def set_workspace_dir(self, workspace_dir):
        """
        Sets Workspace dir
        :param workspace_dir: full path to the workspace dir
        :return:None
        """
        self.workspace_dir = workspace_dir

    def set_mark_as_completed(self, flag):
        """
        This function is used to inform the executing pipeline whether to mark task as completed in the workspace dir or not
        Calling set_mark_as_completed(False) will disable taging task as completed
        so that each time pipeline is run this task will be invoked. This is very useful during debuggin stage where
        usually one would disable marking of the task as completed and only when the code is ready to run in te "production mode"
        the call to set_mark_as_completed(False) woudl get removed so that marking of the task completion is reinstated
        :param flag:boolean flag
        :return:None
        """
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
            full_file_name = osp.abspath(osp.join(self.workspace_dir, file_name))
            file = open(full_file_name, mode)
            return file, full_file_name
        except IOError:
            return None, None

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

            output_file_name = osp.join(self.workspace_dir, rel_file_name)
            output_file_name = osp.abspath(output_file_name)  # normalizing path
            dir_for_output_file_name = osp.dirname(output_file_name)

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
            try:
                dir_name_full_path = osp.abspath(osp.join(self.workspace_dir, dir_name))
                os.makedirs(dir_name_full_path)
                dir_name_dict[dir_name] = dir_name_full_path

            except OSError:
                print('skipping: ' + dir_name_full_path + ' perhaps it already exists')

        return dir_name_dict

    def set_file_resources_to_copy(self, *file_resources, **kwds):
        """
        setter method that allows users to specify file resources and subfolder of the workspace dir
        where the file respirces should be copied
        :param file_resources: arguments list specifying list of files to copy
        :param kwds: dst - destination directory for the move operation - the option supported here
        :return:None
        """

        for file_resource in file_resources:
            try:
                self.file_resources_to_copy[file_resource] = kwds['dst']
            except LookupError:
                self.file_resources_to_copy[file_resource] = ''

    def set_file_resources_to_move(self, *file_resources, **kwds):
        """
        setter method that allows users to specify file resources and subfolder of the workspace dir
        where the file respirces should be moved
        :param file_resources: arguments list specifying list of files to move
        :param kwds: dst - destination directory for the move operation - the option supported here
        :return:None
        """

        for file_resource in file_resources:
            try:
                self.file_resources_to_move[file_resource] = kwds['dst']
            except LookupError:
                self.file_resources_to_move[file_resource] = ''

    def make_dir_tree(self, dirname):
        """
        attempts to create directory tree
        :param dirname: directory name
        :return:None
        """
        import os
        import errno
        try:
            os.makedirs(dirname)
        except OSError, e:
            if e.errno != errno.EEXIST:
                raise IOError('Could not make directory: ' + dirname)

    def copy_file_resources_to_workspace(self):
        """
        Examines dictionary of files to copy and copies listed files to the appropriate workspace folder or its subfolder
        :return:None
        """
        for file_resource, dst_relative_path in self.file_resources_to_copy.items():
            if dst_relative_path != '':
                self.make_dir_tree(os.path.join(self.pipeline.workspace_dir, dst_relative_path))

            file_resource_base_name = os.path.basename(file_resource)
            try:
                target_path = os.path.abspath(
                    os.path.join(self.pipeline.workspace_dir, dst_relative_path, file_resource_base_name))
                shutil.copy(file_resource, target_path)
            except IOError:
                print('Could not copy file: ', file_resource, ' to ', target_path)

    def move_file_resources_to_workspace(self):
        """
        Examines dictionary of files to move and moves listed files to the appropriate workspace folder or its subfolder
        :return:None
        """
        for file_resource, dst_relative_path in self.file_resources_to_move.items():

            if dst_relative_path != '':
                self.make_dir_tree(os.path.join(self.pipeline.workspace_dir, dst_relative_path))

            file_resource_base_name = os.path.basename(file_resource)
            try:
                target_path = os.path.abspath(
                    os.path.join(self.pipeline.workspace_dir, dst_relative_path, file_resource_base_name))
                shutil.move(file_resource, target_path)
            except IOError:
                print('Could not move file: ', file_resource, ' to ', target_path)
            except OSError:
                shutil.copyfile(file_resource, target_path)

    def get_path_to_resource_in_workspace(self, *rel_path_components):
        """
        Returns absolute path to the rel_path_component assuming that rel_path_component is specified w.r.t workspace_dir
        :param rel_path_components: path relative to the workspace dir
        :return:absolute path
        """

        assert self.workspace_dir is not None, "Workspace directory was not set"

        return osp.abspath(osp.join(self.workspace_dir, *rel_path_components))

    def get_pipeline(self):
        """
        Returns pipeline object to which a given task belongs to
        :return:pipeline object
        """
        return self.pipeline

    def get_workspace_dir(self):
        """
        Returns full path to the workspace dir
        :return: full path to the workspace dir
        """
        return self.workspace_dir

    def initialize(self):
        pass

    def restore(self):
        """
        Core function that restores saved results of the task so that following tasks in the pipeline can continue
        :return:
        """
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
        """
        Core function of the task object - needs to be reimplmented in each subslacc deriving from RamTask
        :return:None
        """
        pass


class MatlabRamTask(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.eng = None

    def set_matlab_engine(self, eng):
        """Sets matlab engine - this is done by the pipeline to avoid starting
        of the matlab engine when class inheriting from MatlabRamTask gets
        created. Matlab engines starts are very slow

        :param eng: instance of matlab engine
        :return:None

        """
        self.eng = eng


class TaskRegistry(object):
    """Registry for a series of tasks."""
    def __init__(self):
        self.task_dict = OrderedDict()

    def register_task(self, task):
        task.set_name(type(task).__name__)
        self.task_dict[task.name()] = task

    def run_tasks(self):
        for task_name, task in self.task_dict.items():
            print('RUNNING TASK:', task_name,' obj=', task)
            task.run()
