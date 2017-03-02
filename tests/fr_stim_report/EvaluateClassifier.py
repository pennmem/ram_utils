import ComputeClassifier
from ptsa.data.readers.IndexReader import JsonIndexReader
from os import path
import hashlib
from sklearn.externals import joblib
from ReportTasks.RamTaskMethods import run_lolo_xval,run_loso_xval
from random import shuffle
import numpy as np
from sklearn.metrics import roc_auc_score


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

        task = self.pipeline.task

        experiments = ['FR1','catFR1']+[task]

        for experiment in experiments:
            event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=experiment)))
            for fname in event_files:
                with open(fname,'rb') as f:
                    hash_md5.update(f.read())
        return hash_md5.digest()


    def permuted_loso_AUCs(self, event_sessions, recalls):
        n_perm = self.params.n_perm
        permuted_recalls = np.array(recalls)
        AUCs = np.empty(shape=n_perm, dtype=np.float)
        for i in xrange(n_perm):
            for sess in event_sessions:
                sel = (event_sessions == sess)

                sess_permuted_recalls = permuted_recalls[sel]
                shuffle(sess_permuted_recalls)
                permuted_recalls[sel] = sess_permuted_recalls
            probs = self.lr_classifier.predict_proba(self.pow_mat)[:,0]
            AUCs[i] = roc_auc_score(permuted_recalls,probs)
            print 'AUC =', AUCs[i]
        return AUCs

    def permuted_lolo_AUCs(self, events):
        n_perm = self.params.n_perm
        recalls = events.recalled
        permuted_recalls = np.array(recalls)
        AUCs = np.empty(shape=n_perm, dtype=np.float)
        sessions = np.unique(events.session)
        for i in xrange(n_perm):
            for sess in sessions:
                sess_lists = np.unique(events[events.session==sess].list)
                for lst in sess_lists:
                    sel = (events.session==sess) & (events.list==lst)
                    list_permuted_recalls = permuted_recalls[sel]
                    shuffle(list_permuted_recalls)
                    permuted_recalls[sel] = list_permuted_recalls
            probs = self.lr_classifier.predict_proba(self.pow_mat)[:,0]
            AUCs[i] = roc_auc_score(permuted_recalls, probs)
            print 'AUC =', AUCs[i]
        return AUCs

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
            run_loso_xval(events.session,recalls,self.pow_mat,self.lr_classifier,self.xval_output)

        else:
            print 'Performing in-session permutation test'
            self.perm_AUCs = self.permuted_lolo_AUCs(events)

            print 'Performing leave-one-list-out xval'
            run_lolo_xval(events,recalls,self.pow_mat,self.lr_classifier,self.xval_output)

        self.pvalue = np.sum(self.perm_AUCs >= self.xval_output[-1].auc) / float(self.perm_AUCs.size)

        print 'Perm test p-value = ', self.pvalue
        self.pass_object(task+'_xval_output', self.xval_output)
        self.pass_object(task+'_perm_AUCs', self.perm_AUCs)
        self.pass_object(task+'_pvalue', self.pvalue)

        joblib.dump(self.xval_output, self.get_path_to_resource_in_workspace('-'.join((subject, task, 'xval_output.pkl'))))
        joblib.dump(self.perm_AUCs, self.get_path_to_resource_in_workspace('-'.join((subject, task, 'perm_AUCs.pkl'))))
        joblib.dump(self.pvalue, self.get_path_to_resource_in_workspace('-'.join((subject, task, 'pvalue.pkl'))))

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
