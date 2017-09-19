from os import path
import luigi
import numpy as np
from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader
import hashlib
from RamTaskL import RamTaskL
from FR1EventPreparation import FR1EventPreparation

class RepetitionRatio(RamTaskL):
    recompute_all_ratios = luigi.BoolParameter(default=False)
    repetition_ratios = None
    repetition_percentiles = None


    def define_outputs(self):

        self.add_file_resource('all_repetition_ratios')
        self.add_file_resource('repetition_ratios')
        self.add_file_resource('all_recall_ratios_dict')


    def requires(self):
        yield FR1EventPreparation(pipeline=self.pipeline)


    def input_hashsum(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])

        json_reader = JsonIndexReader(path.join(self.pipeline.mount_point, 'protocols','r1.json'))

        hash_md5 = hashlib.md5()

        event_files = sorted(
            list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
        for fname in event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()


    def run_impl(self):
        subject = self.pipeline.subject.split('_')[0]
        task = self.pipeline.task
        events = self.get_passed_object(task+'_all_events')

        recalls = events[events.recalled == 1]
        sessions = np.unique(recalls.session)
        print '%d sessions' % len(sessions)
        if self.recompute_all_ratios:
            all_recall_ratios_dict = self.initialize_repetition_ratio()
        else:
            try:
                # all_recall_ratios_dict = joblib.load(
                #     path.join(path.basename(self.pipeline.workspace_dir), 'all_repetition_ratios_dict'))
                # all_recall_ratios_dict = self.get_passed_object('all_recall_ratios_dict')

                all_recall_ratios_dict = self.deserialize('all_recall_ratios_dict')

            except IOError:
                all_recall_ratios_dict = self.initialize_repetition_ratio()
        self.pass_object('all_recall_ratios_dict',all_recall_ratios_dict)
        # self.pass_object('all_recall_ratios_dict',all_recall_ratios_dict)
        self.repetition_ratios = all_recall_ratios_dict[subject]

        all_recall_ratios = np.hstack([np.nanmean(x) for x in all_recall_ratios_dict.itervalues()])

        self.pass_object('all_repetition_ratios', all_recall_ratios)
        self.pass_object('repetition_ratios', self.repetition_ratios)

    def get_percentiles(self, all_recall_ratios):
        self.repetition_percentiles = self.repetition_ratios.copy()
        for i, ratio in enumerate(self.repetition_percentiles.flat):
            self.repetition_percentiles.flat[i] = len(np.where(all_recall_ratios < ratio)[0]) / float(
                len(all_recall_ratios.flat))

    def initialize_repetition_ratio(self):
        print self.pipeline.mount_point
        r1 = path.join(self.pipeline.mount_point,'protocols','r1.json')
        print 'r1 location: ',r1
        task = self.pipeline.task
        json_reader = JsonIndexReader(path.join(self.pipeline.mount_point,'protocols','r1.json'))
        subjects = json_reader.subjects(experiment='catFR1')
        all_repetition_rates = {}

        for subject in subjects:
            try:
                print 'Repetition ratios for subject: ', subject

                evs_field_list = ['item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment', 'mstime',
                                  'type','eegoffset', 'recalled', 'item_name', 'intrusion', 'montage','list',
                                  'eegfile', 'msoffset']
                evs_field_list += ['category', 'category_num']

                event_files = sorted(
                    list(json_reader.aggregate_values('all_events', subject=subject,experiment='catFR1')))
                events = None
                for sess_file in event_files:
                    e_path = path.join(str(sess_file))
                    e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)

                    sess_events = e_reader.read()[evs_field_list]

                    if events is None:
                        events = sess_events
                    else:
                        events = np.hstack((events, sess_events))

                events = events.view(np.recarray)
                print len(events), ' events found'
                recalls = events[events.recalled == 1]
                sessions = np.unique(recalls.session)
                lists = np.unique(recalls.list)
                repetition_rates = np.empty([len(sessions), len(lists)])

                for i, r in enumerate(repetition_rates.flat):
                    repetition_rates.flat[i] = np.nan
                for i,session in enumerate(sessions):
                    sess_recalls = recalls[recalls.session == session]
                    lists = np.unique(sess_recalls.list)
                    repetition_rates[i][:len(lists)] = [repetition_ratio(sess_recalls[sess_recalls.list == l])
                                                              for l in lists]
                all_repetition_rates[subject] = repetition_rates.copy()
            except Exception as e:
                print 'Subject ', subject, 'failed:'
                print e
        return all_repetition_rates


def repetition_ratio(recall_list):
    is_repetition = np.diff(recall_list.category_num) == 0
    return np.sum(is_repetition) / float(len(recall_list) - 1)
