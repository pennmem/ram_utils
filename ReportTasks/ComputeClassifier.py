from ReportUtils import ReportRamTask
import numpy as np
from random import shuffle
import warnings
from sklearn.externals import  joblib
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from ptsa.data.readers.IndexReader import JsonIndexReader
import os.path
import itertools

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
        xval_output[-1].compute_roc()
        xval_output[-1].compute_tercile_stats()
        xval_output[-1].compute_normal_approx()

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




