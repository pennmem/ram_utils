import ComputeClassifier
from ptsa.data.readers.IndexReader import JsonIndexReader
from os import path
import hashlib
from sklearn.externals import joblib

class EvaluateClassifier(ComputeClassifier.ComputeClassifier):
    def __init__(self,params,mark_as_completed=False):
        super(EvaluateClassifier,self).__init__(params=params,mark_as_completed=mark_as_completed)

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)
        for fname in bp_paths:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        experiments = ['FR1','catFR1','FR3','catFR3']

        for experiment in experiments:
            event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=experiment)))
            for fname in event_files:
                with open(fname,'rb') as f:
                    hash_md5.update(f.read())
        return hash_md5.digest()

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        self.lr_classifier = self.get_passed_object('lr_classifier')
        events = self.get_passed_object(task+'_events')
        recalls = events.recalled
        self.pow_mat = self.get_passed_object('fr_stim_pow_mat')
        print 'self.pow_mat.shape:',self.pow_mat.shape

        if self.xval_test_type(events) == 'loso':
            print 'Performing permutation test'
            self.perm_AUCs = self.permuted_loso_AUCs(events.session, recalls)

            print 'Performing leave-one-session-out xval'
            self.run_loso_xval(events.session, recalls, permuted=False)
        else:
            print 'Performing in-session permutation test'
            self.perm_AUCs = self.permuted_lolo_AUCs(events)

            print 'Performing leave-one-list-out xval'
            self.run_lolo_xval(events, recalls, permuted=False)


        joblib.dump(self.xval_output, self.get_path_to_resource_in_workspace('-'.join((subject,task, 'xval_output.pkl'))))
        joblib.dump(self.perm_AUCs, self.get_path_to_resource_in_workspace('-'.join((subject,task,'perm_AUCs.pkl'))))
        joblib.dump(self.pvalue, self.get_path_to_resource_in_workspace('-'.join((subject,task,'pvalue.pkl'))))


        self.pass_object(task+'_xval_output', self.xval_output)
        self.pass_object(task+'_perm_AUCs', self.perm_AUCs)
        self.pass_object(task+'_pvalue', self.pvalue)

    def restore(self):

        subject = self.pipeline.subject
        task = self.pipeline.task
        self.xval_output = joblib.load(
                    self.get_path_to_resource_in_workspace('-'.join((subject, task, 'xval_output.pkl'))))
        self.perm_AUCs = joblib.load(self.get_path_to_resource_in_workspace('-'.join((subject, task, 'perm_AUCs.pkl'))))
        self.pvalue = joblib.load(self.get_path_to_resource_in_workspace('-'.join((subject, task, 'pvalue.pkl'))))

        self.pass_object(task + '_xval_output', self.xval_output)
        self.pass_object(task + '_perm_AUCs', self.perm_AUCs)
        self.pass_object(task + '_pvalue', self.pvalue)
