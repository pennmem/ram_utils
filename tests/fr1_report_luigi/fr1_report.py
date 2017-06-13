import luigi
import numpy as np
import os
import os.path
import numpy as np
from sklearn.externals import joblib

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import ReportRamTask

import hashlib
from ReportTasks.RamTaskMethods import create_baseline_events


# # command line example:
# # python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT
#
# # python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=/data10/scratch/mswat/py_run_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT
#
# # python ps_report.py --subject=R1086M --task=FR1 --workspace-dir=/data10/scratch/mswat/R1086M_2 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --matlab-path=~/RAM_UTILS_GIT/tests/ps2_report/AuxiliaryMatlab --python-path=~/RAM_UTILS_GIT
# import sys
# from os.path import *
#
#
# from ReportUtils import CMLParser,ReportPipeline
#
# cml_parser = CMLParser(arg_count_threshold=1)
# cml_parser.arg('--subject','R1304N')
# cml_parser.arg('--task','FR1')
# cml_parser.arg('--workspace-dir','scratch/leond/FR1_reports')
# cml_parser.arg('--mount-point','/Volumes/rhino_root')
# #cml_parser.arg('--recompute-on-no-status')
# # cml_parser.arg('--exit-on-no-change')
#
# args = cml_parser.parse()
#
#
# from FR1EventPreparation import FR1EventPreparation
#
# from RepetitionRatio import RepetitionRatio
#
# from ComputeFR1Powers import ComputeFR1Powers
#
# from MontagePreparation import MontagePreparation
#
# from ComputeFR1HFPowers import ComputeFR1HFPowers
#
# from ComputeTTest import ComputeTTest
#
# from ComputeClassifier import ComputeClassifier,ComputeJointClassifier
#
# from ComposeSessionSummary import ComposeSessionSummary
#
# from GenerateReportTasks import *
#
#
# # turn it into command line options
#
# class Params(object):
#     def __init__(self):
#         self.width = 5
#
#         self.fr1_start_time = 0.0
#         self.fr1_end_time = 1.366
#         self.fr1_buf = 1.365
#
#         self.fr1_retrieval_start_time = -0.525
#         self.fr1_retrieval_end_time = 0.0
#         self.fr1_retrieval_buf = 0.524
#
#         self.hfs_start_time = 0.0
#         self.hfs_end_time = 1.6
#         self.hfs_buf = 1.0
#
#         self.filt_order = 4
#
#         self.freqs = np.logspace(np.log10(6), np.log10(180), 8)
#         self.hfs = np.logspace(np.log10(2), np.log10(200), 50)
#         self.hfs = self.hfs[self.hfs >= 70.0]
#
#         self.encoding_samples_weight = 2.5
#
#         self.log_powers = True
#
#         self.penalty_type = 'l2'
#         self.C = 7.2e-4
#
#         self.n_perm = 200
#         self.parallelize = True





# class pipeline(luigi.Task):
#
#     workspace = luigi.Parameter(default=None)
#     params = luigi.Parameter(default=None)
#     subject = luigi.Parameter(default=None)
#     task = luigi.Parameter(default=None)
#     experiment = luigi.Parameter(default=None)
#     sessions = luigi.Parameter(default=None)
#     exit_on_no_change = luigi.Parameter(default=True)
#     recompute_on_no_status = luigi.Parameter(default=True)
#
#
#     def run(self):
#
#         print self.params
#         print 'HELLO'
#

# class Pipeline(object):
#     workspace_dir = 'd:/scratch/luigi_demo_fr1'
#     params = params
#     mount_point = 'd:/'
#     subject = 'R1065J'
#     # subject = 'R1065J'
#     task = 'FR1'
#     experiment = 'FR1'
#     sessions = None
#     exit_on_no_change = True
#     recompute_on_no_status = True

from Params import Params
from Pipeline import Pipeline
from RamTaskL import RamTaskL

params = Params()
pipeline = Pipeline(params)



# class ReportRamTaskL(luigi.Task):
#     pipeline = luigi.Parameter(default=None)
#     mark_as_completed = luigi.BoolParameter(default=False)
#     file_resources_to_copy = luigi.Parameter(default={})
#
#     def input_hashsum(self):
#         return ''
#
#     def name(self):
#         return self.__class__.__name__
#
#     def add_file_resource(self, name, folder='', ext='pkl', action='copy'):
#         self.file_resources_to_copy[name] = luigi.LocalTarget(
#             join(self.pipeline.workspace_dir, folder, name + '.' + ext.replace('.', '')))
#
#     def get_task_completed_file_name(self):
#         """
#         retunrs name of the task
#         :param task: task object object derived from RamTask or MatlabRamTask
#         :return: task name - this is the name of the derived class
#         """
#         # return join(self.workspace_dir, self.name() + '.completed')
#         return join(self.pipeline.workspace_dir, self.name() + '.completed')
#
#     def is_completed(self):
#         """
#         returns flag indicating if the task was completed or not
#         :param task: task object - object derived from RamTask or MatlabRamTask
#         :return: bool indicating if the file marking the completeion of the task is present
#         and if the dependency hashsum stored in it is equal to the current dependency hashsum
#         """
#         completed_file_name = self.get_task_completed_file_name()
#         if isfile(completed_file_name):
#             f = open(completed_file_name, 'rb')
#             hs = f.read()
#             f.close()
#             return hs == self.input_hashsum()
#         else:
#             return False
#
#     def get_workspace_dir(self):
#         """
#         Returns full path to the workspace dir
#         :return: full path to the workspace dir
#         """
#         return self.pipeline.workspace_dir
#
#     def create_file_in_workspace_dir(self, file_name, mode='w'):
#         """
#         Creates file in the workspace directory - the default file open mode is 'w'.
#         In case certain elements of the path are missing they will be created.
#         :param file_name: file name relative to the workspace directory
#         :param mode: file open mode
#         :return: (file object, full_path_to the file)
#         """
#
#         file_name_to_file_obj_full_path_dict = self.create_multiple_files_in_workspace_dir(file_name, mode=mode)
#
#         try:
#             return file_name_to_file_obj_full_path_dict[file_name]  # returns a tuple (file object, full file name)
#         except LookupError:
#             raise LookupError('Could not locate file_name: %s  in the dictionary of created files' % file_name)
#
#     def create_multiple_files_in_workspace_dir(self, *rel_file_names, **options):
#         """
#         Creates multiple file names in the workspace
#         :param rel_file_names: comma-separated list of file names relative to the workspacedir
#         :param options:
#         default option is mode = 'w'. Other options can be specified using mode='file mode'
#         :return: dictionary {relative_file_path:(file object, full_path_to_created_file)}
#         """
#
#         assert self.pipeline.workspace_dir is not None, "Workspace directory was not set"
#
#         try:
#             mode = options['mode']
#         except LookupError:
#             mode = 'w'
#
#         file_name_to_file_obj_full_path_dict = {}
#
#         for rel_file_name in rel_file_names:
#
#             output_file_name = join(self.pipeline.workspace_dir, rel_file_name)
#             output_file_name = abspath(output_file_name)  # normalizing path
#             dir_for_output_file_name = dirname(output_file_name)
#
#             try:
#                 mkpath(dir_for_output_file_name)
#             except:
#                 raise IOError('Could not create directory path %s' % dir_for_output_file_name)
#
#             try:
#                 file_name_to_file_obj_full_path_dict[rel_file_name] = (open(output_file_name, mode), output_file_name)
#                 # return open(output_file_name, mode),output_file_name
#             except IOError:
#                 raise IOError('COULD NOT OPEN ' + output_file_name + ' in mode=' + mode)
#
#         return file_name_to_file_obj_full_path_dict
#
#     def create_dir_in_workspace(self, dir_name):
#         """
#         Creates directory in the workspace using
#         :param dir_name: directory path relative to the workspace_dir
#         :return: full path to the created directory
#         """
#
#         dir_name_dict = self.create_multiple_dirs_in_workspace(dir_name)
#         try:
#             return dir_name_dict[dir_name]
#         except LookupError:
#             return None
#
#     def create_multiple_dirs_in_workspace(self, *dir_names):
#         """
#         Creates multiple directories in the workspace
#         :param dir_names: comma separated list of the directory paths relative to the workspace_dir
#         :return: dictionary {relative_dir_name:full_path_to_created_dir}
#         """
#
#         assert self.pipeline.workspace_dir is not None, "Workspace directory was not set"
#         dir_name_dict = {}
#         for dir_name in dir_names:
#             # print dir_name
#             try:
#                 dir_name_full_path = abspath(join(self.pipeline.workspace_dir, dir_name))
#                 os.makedirs(dir_name_full_path)
#                 dir_name_dict[dir_name] = dir_name_full_path
#
#             except OSError:
#                 print 'skipping: ' + dir_name_full_path + ' perhaps it already exists'
#                 pass
#
#         return dir_name_dict
#
#     def remove_outputs(self):
#         output_container = self.output()
#
#         itr = None
#
#         def itr_dict(d):
#             for k, v in d.items():
#                 yield v
#
#         def itr_list(l):
#             for v in l:
#                 yield v
#
#         def itr_simple_obj(o):
#             yield o
#
#         if isinstance(output_container, dict):
#             itr = itr_dict(output_container)
#         elif isinstance(output_container, list):
#             itr = itr_list(output_container)
#         else:
#             itr = itr_simple_obj(output_container)
#
#         for output_target in itr:
#             if output_target.exists():
#                 output_target.remove()
#
#     def run_impl(self):
#         pass
#
#     def pass_object(self, name, obj):
#
#         joblib.dump(obj, self.output()[name].path)
#
#         # obj_local = joblib.load(self.output()[name].path)
#         # print obj_local
#
#     def get_passed_object(self, name):
#
#         return joblib.load(self.input()[0][name].path)
#
#     def serialize(self, name, obj):
#
#         with self.output()[name].open('wb') as obj_out:
#             joblib.dump(obj, obj_out)
#
#     def run(self):
#
#         if not self.is_completed():
#             self.remove_outputs()
#             # super(ReportRamTaskL, self).run()
#             self.run_impl()
#             if self.mark_as_completed:
#                 hs = self.input_hashsum()
#                 try:
#                     task_completed_file_name = self.get_task_completed_file_name()
#                     with open(task_completed_file_name, 'wb') as f:
#                         f.write(hs)
#                 except:
#                     print 'No .completed file found'
#                     self.create_file_in_workspace_dir(task_completed_file_name, 'w')
#                     with open(task_completed_file_name, 'wb') as f:
#                         f.write(hs)


class FR1EventPreparation(RamTaskL):

    # def output(self):
    #     out_dict = {}
    #     out_dict[self.pipeline.task+'_all_events'] = luigi.LocalTarget(join(self.pipeline.workspace_dir, self.pipeline.task+'_all_events.pkl'))
    #     out_dict['event_files'] = luigi.LocalTarget(join(self.pipeline.workspace_dir,'event_files.txt'))
    #     out_dict[self.pipeline.task+'_events'] = luigi.LocalTarget(join(self.pipeline.workspace_dir, self.pipeline.task+'_events.pkl'))
    #     out_dict[self.pipeline.task+'_math_events'] = luigi.LocalTarget(join(self.pipeline.workspace_dir, self.pipeline.task+'_math_events.pkl'))
    #     out_dict[self.pipeline.task+'_intr_events'] = luigi.LocalTarget(join(self.pipeline.workspace_dir, self.pipeline.task+'_intr_events.pkl'))
    #     out_dict[self.pipeline.task+'_rec_events'] = luigi.LocalTarget(join(self.pipeline.workspace_dir, self.pipeline.task+'_rec_events.pkl'))
    # 
    #     return out_dict


    # def define_outputs(self):
    #
    #     task = self.pipeline.task
    #     # self.add_file_resource('event_files')
    #     self.add_file_resource(task + '_events')
    #     self.add_file_resource(task + '_all_events')
    #     self.add_file_resource(task + '_math_events')
    #     self.add_file_resource(task + '_intr_events')
    #     self.add_file_resource(task + '_rec_events')

    # def output(self):
    #
    #     # self.define_outputs()
    #     # return self.file_resources_to_copy
    #
    #     out_dict = {}
    #     out_dict[self.pipeline.task+'_all_events'] = luigi.LocalTarget(join(self.pipeline.workspace_dir, self.pipeline.task+'_all_events.pkl'))
    #     # out_dict['event_files'] = luigi.LocalTarget(join(self.pipeline.workspace_dir,'event_files.pkl'))
    #     out_dict[self.pipeline.task+'_events'] = luigi.LocalTarget(join(self.pipeline.workspace_dir, self.pipeline.task+'_events.pkl'))
    #     out_dict[self.pipeline.task+'_math_events'] = luigi.LocalTarget(join(self.pipeline.workspace_dir, self.pipeline.task+'_math_events.pkl'))
    #     out_dict[self.pipeline.task+'_intr_events'] = luigi.LocalTarget(join(self.pipeline.workspace_dir, self.pipeline.task+'_intr_events.pkl'))
    #     out_dict[self.pipeline.task+'_rec_events'] = luigi.LocalTarget(join(self.pipeline.workspace_dir, self.pipeline.task+'_rec_events.pkl'))
    #
    #     return out_dict

        # return self.file_resources_to_copy

    def define_outputs(self):

        task = self.pipeline.task
        # self.add_file_resource('event_files')
        self.add_file_resource(task + '_events')
        self.add_file_resource(task + '_all_events')
        self.add_file_resource(task + '_math_events')
        self.add_file_resource(task + '_intr_events')
        self.add_file_resource(task + '_rec_events')

        # return self.file_resources_to_copy

    def input_hashsum(self):

        subject = self.pipeline.subject
        task = self.pipeline.task
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols', 'r1.json'))

        hash_md5 = hashlib.md5()

        event_files = sorted(
            list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
        for fname in event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def run_impl(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        evs_field_list = ['item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment', 'mstime', 'type',
                          'eegoffset', 'iscorrect', 'answer', 'recalled', 'item_name', 'intrusion', 'montage', 'list',
                          'eegfile', 'msoffset']
        if task == 'catFR1':
            evs_field_list += ['category', 'category_num']

        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols', 'r1.json'))

        if self.pipeline.sessions is None:
            event_files = sorted(
                list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
        else:
            event_files = [json_reader.get_value('all_events', subject=subj_code,
                                                 montage=montage, experiment=task, session=sess)
                           for sess in sorted(self.pipeline.sessions)]

        print 'event_files=', event_files

        # if self.output()['event_files'].exists():
        #     print 'EXISTS'
        # with self.output()['event_files'].open('w') as ev_files_out:
        #     ev_files_out.write('%s' % str(event_files))

        # with self.output()[self.pipeline.task+'_all_events'].open('w') as all_events_out:
        #     all_events_out.write('DUPA')

        # self.serialize(self.pipeline.task+'_all_events','DUPA')

        # with self.output().open('w') as ev_files_out:
        #     ev_files_out.write('%s'%str(event_files))



        events = None
        for sess_file in event_files:
            e_path = os.path.join(self.pipeline.mount_point, str(sess_file))
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

            sess_events = e_reader.read()[evs_field_list]

            if events is None:
                events = sess_events
            else:
                events = np.hstack((events, sess_events))

        events = events.view(np.recarray)

        self.pass_object(task + '_all_events', events)

        events = create_baseline_events(events, start_buffer=1000, end_buffer=29000)

        math_events = events[events.type == 'PROB']

        rec_events = events[events.type == 'REC_WORD']

        intr_events = rec_events[(rec_events.intrusion != -999) & (rec_events.intrusion != 0)]
        irts = np.append([0], np.diff(events.mstime))

        events = events[
            (events.type == 'WORD') | ((events.intrusion == 0) & (irts > 1000)) | (events.type == 'REC_BASE')]

        print len(events), task, 'WORD events'

        self.pass_object(task + '_events', events)
        self.pass_object(task + '_math_events', math_events)
        self.pass_object(task + '_intr_events', intr_events)
        self.pass_object(task + '_rec_events', rec_events)


class EventCheck(RamTaskL):
    def requires(self):
        yield FR1EventPreparation(pipeline=pipeline, mark_as_completed=True)

    # def requires(self):
    #     FR1EventPreparation(pipeline=pipeline, mark_as_completed=True)



    def run_impl(self):
        print 'GOT HERE'

        # with self.input()[0]['event_files'].open('r') as f:
        #     print 'THOSE ARE READ FILES ', f.read()

        events = self.get_passed_object(self.pipeline.task + '_all_events')
        print events

        #
        # pass


if __name__ == '__main__':

    luigi.build([FR1EventPreparation(pipeline=pipeline, mark_as_completed=True), EventCheck(pipeline=pipeline)],
                local_scheduler=True)


    # luigi.build([pipeline(params=params,subject='R1065J',task='FR1'), FR1EventPreparation()], local_scheduler=True)

    # luigi.build([FR1EventPreparation(pipeline=pipeline, mark_as_completed=True), EventCheck(pipeline=pipeline)],
    #             local_scheduler=True)
    # luigi.build([EventCheck(pipeline=pipeline)], local_scheduler=True)

#
#
# params = Params()
#
#
#
# # sets up processing pipeline
# report_pipeline = ReportPipeline(subject=args.subject, task=args.task,experiment=args.task,sessions =args.sessions,
#                                  workspace_dir=join(args.workspace_dir,args.subject), mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change,
#                                  recompute_on_no_status=args.recompute_on_no_status)
#
#
# report_pipeline.add_task(FR1EventPreparation(mark_as_completed=False))
#
# report_pipeline.add_task(MontagePreparation(params, mark_as_completed=False))
#
# if 'cat' in args.task:
#     report_pipeline.add_task(RepetitionRatio(mark_as_completed=True))
#
# report_pipeline.add_task(ComputeFR1Powers(params=params, mark_as_completed=True))
#
# report_pipeline.add_task(ComputeFR1HFPowers(params=params, mark_as_completed=True))
#
# report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=True))
#
# report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=False))
#
# report_pipeline.add_task(ComputeJointClassifier(params=params,mark_as_completed=False))
#
# report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))
#
# report_pipeline.add_task(GeneratePlots(mark_as_completed=False))
#
# report_pipeline.add_task(GenerateTex(mark_as_completed=False))
#
# report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))
#
#
# # starts processing pipeline
# report_pipeline.execute_pipeline()
