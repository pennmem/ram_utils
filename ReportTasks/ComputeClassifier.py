from ReportUtils import ReportRamTask
import numpy as np
from random import shuffle
import warnings
from sklearn.externals import  joblib
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression
from ptsa.data.readers.IndexReader import JsonIndexReader
import os.path
import itertools
from math import sqrt



class ModelOutput(object):
    def __init__(self, true_labels, probs):
        self.true_labels = np.array(true_labels)
        self.probs = np.array(probs)
        self.auc = np.nan
        self.fpr = np.nan
        self.tpr = np.nan
        self.thresholds = np.nan
        self.jstat_thresh = np.nan
        self.jstat_quantile = np.nan
        self.low_pc_diff_from_mean = np.nan
        self.mid_pc_diff_from_mean = np.nan
        self.high_pc_diff_from_mean = np.nan
        self.n1 = np.nan
        self.mean1 = np.nan
        #self.std1 = np.nan
        self.n0 = np.nan
        self.mean0 = np.nan
        #self.std0 = np.nan
        self.pooled_std = np.nan

    def compute_normal_approx(self):
        class1_mask = (self.true_labels==1)
        class1_probs = self.probs[class1_mask]
        self.n1 = len(class1_probs)
        class1_normal = np.log(class1_probs/(1.0-class1_probs))
        self.mean1 = np.mean(class1_normal)
        #self.std1 = np.std(class1_normal, ddof=1)
        var1 = np.var(class1_normal, ddof=1)
        print 'Positive class: mean =', self.mean1, 'variance =', var1, 'n =', self.n1

        class0_probs = self.probs[~class1_mask]
        self.n0 = len(class0_probs)
        class0_normal = np.log(class0_probs/(1.0-class0_probs))
        self.mean0 = np.mean(class0_normal)
        #self.std0 = np.std(class0_normal, ddof=1)
        var0 = np.var(class0_normal, ddof=1)
        print 'Negative class: mean =', self.mean0, 'variance =', var0, 'n =', self.n0

        self.pooled_std = sqrt((var1*(self.n1-1) + var0*(self.n0-1)) / (self.n1+self.n0-2))

        #if self.mean1 < self.mean0:
        #    print 'BAD CLASSIFIER: recall class mean is less than non-recall class mean!!'
        #    sys.exit(0)

    def compute_roc(self):
        try:
            self.auc = roc_auc_score(self.true_labels, self.probs)
        except ValueError:
            return
        self.fpr, self.tpr, self.thresholds = roc_curve(self.true_labels, self.probs)
        self.jstat_quantile = 0.5
        self.jstat_thresh = np.median(self.probs)

    def compute_tercile_stats(self):
        thresh_low = np.percentile(self.probs, 100.0/3.0)
        thresh_high = np.percentile(self.probs, 2.0*100.0/3.0)

        low_terc_sel = (self.probs <= thresh_low)
        high_terc_sel = (self.probs >= thresh_high)
        mid_terc_sel = ~(low_terc_sel | high_terc_sel)

        low_terc_recall_rate = np.sum(self.true_labels[low_terc_sel]) / float(np.sum(low_terc_sel))
        mid_terc_recall_rate = np.sum(self.true_labels[mid_terc_sel]) / float(np.sum(mid_terc_sel))
        high_terc_recall_rate = np.sum(self.true_labels[high_terc_sel]) / float(np.sum(high_terc_sel))

        recall_rate = np.sum(self.true_labels) / float(self.true_labels.size)

        self.low_pc_diff_from_mean = 100.0 * (low_terc_recall_rate-recall_rate) / recall_rate
        self.mid_pc_diff_from_mean = 100.0 * (mid_terc_recall_rate-recall_rate) / recall_rate
        self.high_pc_diff_from_mean = 100.0 * (high_terc_recall_rate-recall_rate) / recall_rate


class ComputeClassifier(ReportRamTask):
    def __init__(self,params,mark_as_completed):
        super(ReportRamTask,self).__init__(mark_as_completed)
        self.params=params
        self.lr_classifier = LogisticRegression(C=params.C,class_weight=params.class_weights)
        self.pow_mat = np.zeros((0,))
        self.xval_output={}

    def input_hashsum(self):
        subject = self.pipeline.subject.split['_']
        montage = 0 if len(subject)==1 else subject[1]
        subject=subject[0]

        tasks = self.params.tasks

        jr = JsonIndexReader(os.path.join(self.pipeline.mount_point,'protocols','r1.json'))
        events = list(itertools.chain(
            *[jr.aggregate_values('task_events',subject=subject,experiment=task,montage=montage) for task in tasks]
        ))
        for path in events:
            with open(path,'w') as event:
                self.hash.update(event)
        pair_files = jr.aggregate_values('pairs',subject=subject,experiment=task,montage=montage)
        for path in pair_files:
            with open(path,'w') as pair_file:
                self.hash.update(pair_file)

        return self.hash.digest()


    @staticmethod
    def xval_test_type(events):
        event_sessions = events.session
        sessions = np.unique(event_sessions)
        if len(sessions) == 1:
            return 'lolo'
        for sess in sessions:
            sess_events = events[event_sessions == sess]
            if len(sess_events) >= 0.7 * len(events):
                return 'lolo'
            return 'loso'

    @staticmethod
    def recalls(events):
        return events.recalled

    def permuted_aucs(self,events):
        n_perm = self.params.n_perm
        recalls = self.recalls(events)
        AUCs = np.empty(shape=n_perm, dtype=np.float)
        with joblib.Parallel(n_jobs=-1,verbose=20) as parallel:
            probs = parallel( joblib.delayed(self.xval_function)(events,recalls,self.lr_classifier,self.pow_mat,self.xval_output,permuted=True)
                              for i in range(n_perm))
            AUCs[:] = [roc_auc_score(recalls, p) for p in probs]
        return AUCs

    @property
    def events(self):
        return self.get_passed_object('events')

    def pow_mat(self):
        return self.get_passed_object('pow_mat')


    def run(self):
        subject = self.pipeline.subject

        self.xval_function = run_lolo_xval if self.xval_test_type(self.events) == 'lolo' else run_loso_xval

        print 'Performing permutation test'

        self.perm_AUCs = self.permuted_aucs(self.events)

        print 'Performing %s cross-validation'%self.xval_test_type(self.events)
        self.xval_function(self.events,self.recalls(self.events),self.lr_classifier,self.pow_mat,self.xval_output)

        self.pvalue = np.sum(self.perm_AUCs >= self.xval_output[-1].auc) / float(self.perm_AUCs.size)
        print 'AUC =', self.xval_output[-1].auc

        self.pvalue = np.sum(self.perm_AUCs >= self.xval_output[-1].auc) / float(self.perm_AUCs.size)
        print 'Perm test p-value =', self.pvalue

        print 'thresh =', self.xval_output[-1].jstat_thresh, 'quantile =', self.xval_output[-1].jstat_quantile

        # Finally, fitting classifier on all available data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lr_classifier.fit(self.pow_mat, self.recalls(self.events))

        self.pass_object('lr_classifier', self.lr_classifier)
        self.pass_object('xval_output', self.xval_output)
        self.pass_object('perm_AUCs', self.perm_AUCs)
        self.pass_object('pvalue', self.pvalue)

        joblib.dump(self.lr_classifier, self.get_path_to_resource_in_workspace(subject + '-lr_classifier.pkl'))
        joblib.dump(self.xval_output, self.get_path_to_resource_in_workspace(subject + '-xval_output.pkl'))
        joblib.dump(self.perm_AUCs, self.get_path_to_resource_in_workspace(subject + '-perm_AUCs.pkl'))
        joblib.dump(self.pvalue, self.get_path_to_resource_in_workspace(subject + '-pvalue.pkl'))

class ComputePALClassifier(ComputeClassifier):
    @staticmethod
    def recalls(events):
        return events.correct


def run_lolo_xval(events,recalls,classifier,pow_mat,xval_output,permuted=False):
    # (np.recarray,np.array[bool],LogisticRegression, np.array, dict[int,ModelOutput],bool) -> np.array(float)
    """
    :param events:
    :param recalls:
    :param classifier:
    :param pow_mat:
    :param xval_output:
    :param permuted:
    :return:
        probabilities
    """
    probs = np.empty_like(recalls, dtype=np.float)
    permuted_recalls = np.array(recalls)
    sessions = np.unique(events.session)
    if permuted:
        for sess in sessions:
            sess_lists = np.unique(events[events.session == sess].list)
            for lst in sess_lists:
                sel = (events.session == sess) & (events.list == lst)
                list_permuted_recalls = permuted_recalls[sel]
                shuffle(list_permuted_recalls)
                permuted_recalls[sel] = list_permuted_recalls
                insample_mask = (events.session != sess) | (events.list != lst)
                insample_pow_mat = pow_mat[insample_mask]
                insample_recalls = recalls[insample_mask]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    classifier.fit(insample_pow_mat, insample_recalls)

                outsample_mask = ~insample_mask
                outsample_pow_mat = pow_mat[outsample_mask]

                probs[outsample_mask] = classifier.predict_proba(outsample_pow_mat)[:, 1]

    if not permuted:
        mo = ModelOutput(recalls,probs)
        mo.compute_roc()
        mo.compute_tercile_stats()
        mo.compute_normal_approx()
        xval_output[-1] = mo

    return probs


def run_loso_xval(events,recalls,classifier,pow_mat,xval_output,permuted=False):
    """
    :param events:
    :param recalls:
    :param classifier:
    :param pow_mat:
    :param xval_output:
    :param permuted:
    :return:
    """
    probs = np.empty_like(recalls, dtype=np.float)
    permuted_recalls = np.array(recalls)
    sessions = np.unique(events.session)
    if permuted:
        for sess in np.unique(events.session):
            sel = (events.session == sess)
            sess_permuted_recalls = permuted_recalls[sel]
            shuffle(sess_permuted_recalls)
            permuted_recalls[sel] = sess_permuted_recalls

            insample_mask = (events.session != sess) | (events.list != lst)
            insample_pow_mat = pow_mat[insample_mask]
            insample_recalls = recalls[insample_mask]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                classifier.fit(insample_pow_mat, insample_recalls)

            outsample_mask = ~insample_mask
            outsample_pow_mat = pow_mat[outsample_mask]

            probs[outsample_mask] = classifier.predict_proba(outsample_pow_mat)[:, 1]

    if not permuted:
        xval_output[-1].compute_roc()
        xval_output[-1].compute_tercile_stats()
        xval_output[-1].compute_normal_approx()

    return probs




