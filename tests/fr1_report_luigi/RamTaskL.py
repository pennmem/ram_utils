import luigi
import numpy as np
import os
import os.path
from os.path import *
import numpy as np
from sklearn.externals import joblib
from collections import defaultdict

class RamTaskL(luigi.Task):
    pipeline = luigi.Parameter(default=None)
    mark_as_completed = luigi.BoolParameter(default=False)
    # file_resources_to_copy = luigi.Parameter(default={})
    file_resources_to_copy = defaultdict(dict)

    def input_hashsum(self):
        return ''

    def name(self):
        return self.__class__.__name__

    def add_file_resource(self, name, folder='', ext='pkl', action='copy'):
        if folder == '' or folder == self.__class__.__name__:
            folder_tmp = self.__class__.__name__
        elif folder.strip() == '.':
            folder_tmp = ''
        else:
            folder_tmp = folder


        self.file_resources_to_copy[self.__class__.__name__][name] = luigi.LocalTarget(
            join(self.pipeline.workspace_dir, folder_tmp, name + '.' + ext.replace('.', '')))

    def get_task_completed_file_name(self):
        """
        retunrs name of the task
        :param task: task object object derived from RamTask or MatlabRamTask
        :return: task name - this is the name of the derived class
        """
        # return join(self.workspace_dir, self.name() + '.completed')
        return join(self.pipeline.workspace_dir, self.name() + '.completed')

    def is_completed(self):
        """
        returns flag indicating if the task was completed or not
        :param task: task object - object derived from RamTask or MatlabRamTask
        :return: bool indicating if the file marking the completeion of the task is present
        and if the dependency hashsum stored in it is equal to the current dependency hashsum
        """
        completed_file_name = self.get_task_completed_file_name()
        if isfile(completed_file_name):
            f = open(completed_file_name, 'rb')
            hs = f.read()
            f.close()
            return hs == self.input_hashsum()
        else:
            return False

    @property
    def workspace_dir(self):
        return self.pipeline.workspace_dir

    def get_workspace_dir(self):
        """
        Returns full path to the workspace dir
        :return: full path to the workspace dir
        """
        return self.workspace_dir
        # return self.pipeline.workspace_dir

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

        assert self.pipeline.workspace_dir is not None, "Workspace directory was not set"

        try:
            mode = options['mode']
        except LookupError:
            mode = 'w'

        file_name_to_file_obj_full_path_dict = {}

        for rel_file_name in rel_file_names:

            output_file_name = join(self.pipeline.workspace_dir, rel_file_name)
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

        assert self.pipeline.workspace_dir is not None, "Workspace directory was not set"
        dir_name_dict = {}
        for dir_name in dir_names:
            # print dir_name
            try:
                dir_name_full_path = abspath(join(self.pipeline.workspace_dir, dir_name))
                os.makedirs(dir_name_full_path)
                dir_name_dict[dir_name] = dir_name_full_path

            except OSError:
                print 'skipping: ' + dir_name_full_path + ' perhaps it already exists'
                pass

        return dir_name_dict

    def get_path_to_resource_in_workspace(self, *rel_path_components):
        """
        Returns absolute path to the rel_path_component assuming that rel_path_component is specified w.r.t workspace_dir
        :param rel_path_components: path relative to the workspace dir
        :return:absolute path
        """

        assert self.workspace_dir is not None, "Workspace directory was not set"

        return abspath(join(self.workspace_dir, *rel_path_components))



    def run_impl(self):
        pass

    def serialize(self, name, obj):
        joblib.dump(obj, join(self.pipeline.workspace_dir,name,'.pkl'))

    def deserialize(self, name):
        return joblib.load(join(self.pipeline.workspace_dir,name,'.pkl'))


    def pass_object(self, name, obj, serialize=True):
        # blanking the file

        self.pipeline.passed_objects_dict[name] = obj
        if serialize:

            with self.output()[name].open('w'):
                pass

            joblib.dump(obj, self.output()[name].path)

    def get_passed_object(self, name):
        try:
            return self.pipeline.passed_objects_dict[name]
        except KeyError:
            obj_list = []
            for input_dict in self.input():

                try:
                    obj = joblib.load(input_dict[name].path)
                    obj_list.append(obj)
                except KeyError:
                    pass
            if len(obj_list)>1:
                raise KeyError('Found multiple %s in the inputs. This is illegal in RamTasks'%name)
            elif len(obj_list)==0:
                raise KeyError('Could not find %s in any of the inputs'%name)


            self.pipeline.passed_objects_dict[name] = obj_list[0]

            return obj_list[0]




#
    # def serialize(self, name, obj):
    #
    #
    #     with self.output()[name].open('wb') as obj_out:
    #         joblib.dump(obj, obj_out)

    def define_outputs(self):
        pass

    def output(self):

        self.define_outputs()
        if self.mark_as_completed:
            self.add_file_resource(self.__class__.__name__,  folder='.', ext='completed', action='copy')

        try:
            return self.file_resources_to_copy[self.__class__.__name__]
        except KeyError:
            return None


    def remove_outputs(self):

        output_container = self.output()

        itr = None

        def itr_dict(d):
            for k, v in d.items():
                yield v

        def itr_list(l):
            for v in l:
                yield v

        def itr_simple_obj(o):
            yield o

        if isinstance(output_container, dict):
            itr = itr_dict(output_container)
        elif isinstance(output_container, list):
            itr = itr_list(output_container)
        else:
            itr = itr_simple_obj(output_container)

        for output_target in itr:
            if output_target.exists():
                output_target.remove()


    def run(self):

        if not self.is_completed():
            self.remove_outputs()
            # super(ReportRamTaskL, self).run()

            self.run_impl()

            if self.mark_as_completed:
                hs = self.input_hashsum()
                try:
                    task_completed_file_name = self.get_task_completed_file_name()
                    with open(task_completed_file_name, 'wb') as f:
                        f.write(hs)
                except:
                    print 'No *.completed file found'
                    self.create_file_in_workspace_dir(task_completed_file_name, 'w')
                    with open(task_completed_file_name, 'wb') as f:
                        f.write(hs)
