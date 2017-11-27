import hashlib
import os
import warnings
from random import shuffle

import numpy as np
from ReportTasks.RamTaskMethods import (
    run_lolo_xval, run_loso_xval, permuted_loso_AUCs, permuted_lolo_AUCs,
    ModelOutput
)
from ptsa.data.readers.IndexReader import JsonIndexReader
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from ramutils.classifier.weighting import get_sample_weights
from ramutils.pipeline import RamTask
from ramutils.powers import normalize_powers_by_session

try:
    from typing import Dict
except ImportError:
    pass


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

    def compute_roc(self):
        try:
            self.auc = roc_auc_score(self.true_labels, self.probs)
        except ValueError:
            return
        self.fpr, self.tpr, self.thresholds = roc_curve(self.true_labels, self.probs)
        self.jstat_quantile = 0.5
        self.jstat_thresh = np.median(self.probs)

    def compute_tercile_stats(self):
        thresh_low = np.percentile(self.probs, 100.0 / 3.0)
        thresh_high = np.percentile(self.probs, 2.0 * 100.0 / 3.0)

        low_terc_sel = (self.probs <= thresh_low)
        high_terc_sel = (self.probs >= thresh_high)
        mid_terc_sel = ~(low_terc_sel | high_terc_sel)

        low_terc_recall_rate = np.sum(self.true_labels[low_terc_sel]) / float(np.sum(low_terc_sel))
        mid_terc_recall_rate = np.sum(self.true_labels[mid_terc_sel]) / float(np.sum(mid_terc_sel))
        high_terc_recall_rate = np.sum(self.true_labels[high_terc_sel]) / float(np.sum(high_terc_sel))

        recall_rate = np.sum(self.true_labels) / float(self.true_labels.size)

        self.low_pc_diff_from_mean = 100.0 * (low_terc_recall_rate - recall_rate) / recall_rate
        self.mid_pc_diff_from_mean = 100.0 * (mid_terc_recall_rate - recall_rate) / recall_rate
        self.high_pc_diff_from_mean = 100.0 * (high_terc_recall_rate - recall_rate) / recall_rate


class ComputeClassifier(RamTask):
    def __init__(self, params, mark_as_completed=True, force_rerun=False):
        RamTask.__init__(self, mark_as_completed, force_rerun)
        self.params = params
        self.pow_mat = None
        self.lr_classifier = None

        # ModelOutput per session; xval_output[-1] is across all sessions
        self.xval_output = dict()  # type: Dict[ModelOutput]

        self.perm_AUCs = None
        self.pvalue = None

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)
        for fname in bp_paths:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def get_auc(self, classifier, features, recalls, mask):

        masked_recalls = recalls[mask]
        probs = classifier.predict_proba(features[mask])[:, 1]
        auc = roc_auc_score(masked_recalls, probs)
        return auc

    def get_pow_mat(self):
        """Filters the whole power matrix by removing excluded pairs.

        This should really be defined as a property...

        """
        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        reduced_pairs = self.get_passed_object('reduced_pairs')
        to_include = np.array([bp in reduced_pairs for bp in bipolar_pairs])
        pow_mat = self.get_passed_object('pow_mat')
        pow_mat = pow_mat.reshape((len(pow_mat), -1, len(self.params.freqs)))[:,to_include,:].reshape((len(pow_mat),-1))
        return pow_mat

    def pass_objects(self):
        subject=self.pipeline.subject
        self.pass_object('lr_classifier', self.lr_classifier)
        self.pass_object('xval_output', self.xval_output)
        self.pass_object('perm_AUCs', self.perm_AUCs)
        self.pass_object('pvalue', self.pvalue)
        self.pass_object('reduced_pow_mat', self.pow_mat)

        classifier_path = self.get_path_to_resource_in_workspace(subject + '-lr_classifier.pkl')
        joblib.dump(self.lr_classifier, classifier_path)
        joblib.dump(self.xval_output, self.get_path_to_resource_in_workspace(subject + '-xval_output.pkl'))
        joblib.dump(self.perm_AUCs, self.get_path_to_resource_in_workspace(subject + '-perm_AUCs.pkl'))
        joblib.dump(self.pvalue, self.get_path_to_resource_in_workspace(subject + '-pvalue.pkl'))
        joblib.dump(self.pow_mat, self.get_path_to_resource_in_workspace(subject + '-reduced_pow_mat.pkl'))

        self.pass_object('classifier_path', classifier_path)

    def run(self):
        subject = self.pipeline.subject
        events = self.get_passed_object('FR_events')
        self.pow_mat = self.get_pow_mat()
        encoding_mask = events.type=='WORD'
        self.pow_mat[encoding_mask] = normalize_powers_by_session(self.pow_mat[encoding_mask], events[encoding_mask])
        self.pow_mat[~encoding_mask] = normalize_powers_by_session(self.pow_mat[~encoding_mask], events[~encoding_mask])

        self.lr_classifier = LogisticRegression(C=self.params.C,
                                                penalty=self.params.penalty_type,
                                                solver='liblinear')

        event_sessions = events.session

        recalls = events.recalled
        recalls[events.type=='REC_WORD'] = 1
        recalls[events.type=='REC_BASE'] = 0

        sample_weights = get_sample_weights(events,
                                            self.params.encoding_samples_weight)

        sessions = np.unique(event_sessions)
        if len(sessions) > 1:
            print 'Performing permutation test'
            self.perm_AUCs = self.permuted_loso_AUCs(event_sessions, recalls, sample_weights,events=events)

            print 'Performing leave-one-session-out xval'
            _,encoding_probs = self.run_loso_xval(event_sessions, recalls, permuted=False,samples_weights=sample_weights, events=events)
            print 'CROSS VALIDATION ENCODING AUC = ', roc_auc_score(events[events.type == 'WORD'].recalled,
                                                                    encoding_probs)
        else:
            sess = sessions[0]
            event_lists = events.list

            print 'Performing in-session permutation test'
            self.perm_AUCs = self.permuted_lolo_AUCs(sess, event_lists, recalls,samples_weights=sample_weights)

            print 'Performing leave-one-list-out xval'
            self.run_lolo_xval(sess, event_lists, recalls, permuted=False,samples_weights=sample_weights)

        print 'CROSS VALIDATION AUC =', self.xval_output[-1].auc

        self.pvalue = np.nansum(self.perm_AUCs >= self.xval_output[-1].auc) / float(self.perm_AUCs[~np.isnan(self.perm_AUCs)].size)
        print 'Perm test p-value =', self.pvalue

        print 'thresh =', self.xval_output[-1].jstat_thresh, 'quantile =', self.xval_output[-1].jstat_quantile

        self.lr_classifier.fit(self.pow_mat, recalls, sample_weights)
        recall_prob_array = self.lr_classifier.predict_proba(self.pow_mat)[:,1]
        insample_auc = roc_auc_score(recalls, recall_prob_array)
        print 'in-sample AUC=', insample_auc

        model_weights = self.lr_classifier.coef_
        # Specify that the file should overwrite so that when
        # ComputeClassifier and ComputeFullClassifier are run back to back,
        # it will not complain about the dataset already existing in the h5 file
        try:
            self.save_array_to_hdf5(self.get_path_to_resource_in_workspace(subject + "-debug_data.h5"),
                                    "model_output",
                                    recall_prob_array,
                                    overwrite=True)
            self.save_array_to_hdf5(self.get_path_to_resource_in_workspace(subject + "-debug_data.h5"),
                                    "model_weights",
                                model_weights)
        except Exception:
            print('could not save debug data')

        self.pass_objects()

    def run_loso_xval(self, event_sessions, recalls, permuted=False,samples_weights=None, events=None):
        """
        Note samples_weights is not really used for computations it is used to only check if it is None i.e. as a flag
         Weird but will leave it for now
        :param event_sessions:
        :param recalls:
        :param permuted:
        :param samples_weights:
        :param events:
        :return:
        """

        probs = np.empty_like(recalls, dtype=np.float)
        if events is not None:
            encoding_probs = np.empty_like(events[events.type=='WORD'],dtype=np.float)

        sessions = np.unique(event_sessions)

        auc_encoding = np.empty(sessions.shape[0], dtype=np.float)
        auc_retrieval = np.empty(sessions.shape[0], dtype=np.float)
        auc_both = np.empty(sessions.shape[0], dtype=np.float)

        for sess_idx, sess in enumerate(sessions):
            insample_mask = (event_sessions != sess)
            insample_pow_mat = self.pow_mat[insample_mask]
            insample_recalls = recalls[insample_mask]
            insample_samples_weights = get_sample_weights(events[events.session != sess],
                                                          self.params.encoding_samples_weight)

            outsample_both_mask = (events.session == sess)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if samples_weights is not None:
                    self.lr_classifier.fit(insample_pow_mat,
                                           insample_recalls,
                                           insample_samples_weights)
                else:
                    self.lr_classifier.fit(insample_pow_mat,
                                           insample_recalls)

            outsample_mask = ~insample_mask
            outsample_pow_mat = self.pow_mat[outsample_mask]
            outsample_recalls = recalls[outsample_mask]

            outsample_probs = self.lr_classifier.predict_proba(outsample_pow_mat)[:, 1]
            if not permuted:
                self.xval_output[sess] = ModelOutput(outsample_recalls, outsample_probs)
                self.xval_output[sess].compute_roc()
                self.xval_output[sess].compute_tercile_stats()
            probs[outsample_mask] = outsample_probs

            if events is not None:
                outsample_encoding_mask = (events.session == sess) & (events.type == 'WORD')
                outsample_retrieval_mask = (events.session == sess) & ((events.type == 'REC_BASE') | (events.type == 'REC_WORD'))
                outsample_both_mask = (events.session == sess)

                auc_encoding[sess_idx] = self.get_auc(
                    classifier=self.lr_classifier, features=self.pow_mat, recalls=recalls, mask=outsample_encoding_mask)
                encoding_probs[events[events.type=='WORD'].session==sess] = self.lr_classifier.predict_proba(self.pow_mat[outsample_encoding_mask])[:,1]

                auc_retrieval[sess_idx] = self.get_auc(
                    classifier=self.lr_classifier, features=self.pow_mat, recalls=recalls, mask=outsample_retrieval_mask)

                auc_both[sess_idx] = self.get_auc(
                    classifier=self.lr_classifier, features=self.pow_mat, recalls=recalls, mask=outsample_both_mask)

        if not permuted:
            self.xval_output[-1] = ModelOutput(recalls[events.type=='WORD'], probs[events.type=='WORD'])
            self.xval_output[-1].compute_roc()
            self.xval_output[-1].compute_tercile_stats()

            print 'auc_encoding=',auc_encoding, np.mean(auc_encoding)
            print 'auc_retrieval=',auc_retrieval, np.mean(auc_retrieval)
            print 'auc_both=',auc_both, np.mean(auc_both)

        if events is None:
            return probs
        else:
            return (probs, encoding_probs)

    def permuted_loso_AUCs(self, event_sessions, recalls, samples_weights=None,events=None):
        n_perm = self.params.n_perm
        permuted_recalls = np.array(recalls)
        AUCs = np.empty(shape=n_perm, dtype=np.float)
        for i in xrange(n_perm):
            try:
                for sess in np.unique(event_sessions):
                    sel = (event_sessions == sess)
                    sess_permuted_recalls = permuted_recalls[sel]
                    shuffle(sess_permuted_recalls)
                    permuted_recalls[sel] = sess_permuted_recalls
                probs = self.run_loso_xval(event_sessions, permuted_recalls, permuted=True,samples_weights=samples_weights,events=events)[0]
                AUCs[i] = roc_auc_score(recalls, probs)
                print 'AUC =', AUCs[i]
            except ValueError:
                AUCs[i] = np.nan
        return AUCs

    def run_lolo_xval(self, sess, event_lists, recalls, permuted=False, samples_weights=None):
        probs = np.empty_like(recalls, dtype=np.float)

        lists = np.unique(event_lists)

        for lst in lists:
            insample_mask = (event_lists != lst)
            insample_pow_mat = self.pow_mat[insample_mask]
            insample_recalls = recalls[insample_mask]
            insample_samples_weights = samples_weights[insample_mask]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if samples_weights is not None:
                    self.lr_classifier.fit(insample_pow_mat, insample_recalls,insample_samples_weights)
                else:
                    self.lr_classifier.fit(insample_pow_mat, insample_recalls)

            outsample_mask = ~insample_mask
            outsample_pow_mat = self.pow_mat[outsample_mask]

            probs[outsample_mask] = self.lr_classifier.predict_proba(outsample_pow_mat)[:, 1]

        if not permuted:
            xval_output = ModelOutput(recalls, probs)
            xval_output.compute_roc()
            xval_output.compute_tercile_stats()
            self.xval_output[sess] = self.xval_output[-1] = xval_output

        return probs

    def permuted_lolo_AUCs(self, sess, event_lists, recalls,samples_weights=None):
        n_perm = self.params.n_perm
        permuted_recalls = np.array(recalls)
        AUCs = np.empty(shape=n_perm, dtype=np.float)
        for i in xrange(n_perm):
            for lst in event_lists:
                sel = (event_lists == lst)
                list_permuted_recalls = permuted_recalls[sel]
                shuffle(list_permuted_recalls)
                permuted_recalls[sel] = list_permuted_recalls
            probs = self.run_lolo_xval(sess, event_lists, permuted_recalls, permuted=True,samples_weights=samples_weights)
            AUCs[i] = roc_auc_score(recalls, probs)
            print 'AUC =', AUCs[i]
        return AUCs

    def restore(self):
        subject = self.pipeline.subject

        classifier_path = self.get_path_to_resource_in_workspace(subject + '-lr_classifier.pkl')
        self.lr_classifier = joblib.load(classifier_path)
        # self.lr_classifier = joblib.load(self.get_path_to_resource_in_workspace(subject + '-lr_classifier.pkl'))
        self.xval_output = joblib.load(self.get_path_to_resource_in_workspace(subject + '-xval_output.pkl'))
        self.perm_AUCs = joblib.load(self.get_path_to_resource_in_workspace(subject + '-perm_AUCs.pkl'))
        self.pvalue = joblib.load(self.get_path_to_resource_in_workspace(subject + '-pvalue.pkl'))

        self.pass_object('classifier_path', classifier_path)
        self.pass_object('lr_classifier', self.lr_classifier)
        self.pass_object('xval_output', self.xval_output)
        self.pass_object('perm_AUCs', self.perm_AUCs)
        self.pass_object('pvalue', self.pvalue)


class ComputeFullClassifier(ComputeClassifier):
    def get_pow_mat(self):
        return self.get_passed_object('pow_mat')

    def restore(self):
        subject=self.pipeline.subject
        full_classifier_path = self.get_path_to_resource_in_workspace(subject+'-xval_output_all_electrodes.pkl')
        self.xval_output = joblib.load(full_classifier_path)
        self.compare_AUCs()
        self.pass_object('full_classifier_path',full_classifier_path)
        self.pass_object('xval_output_all_electrodes',self.xval_output)

    def pass_objects(self):
        subject=self.pipeline.subject
        classifier_path = self.get_path_to_resource_in_workspace(subject+'lr_classifier_full.pkl')
        joblib.dump(self.lr_classifier,classifier_path)
        joblib.dump(self.xval_output,self.get_path_to_resource_in_workspace(subject+'-xval_output_all_electrodes.pkl'))
        self.pass_object('full_classifier_path',classifier_path)
        self.pass_object('xval_output_all_electrodes',self.xval_output)

    def compare_AUCs(self):
        reduced_xval_output = self.get_passed_object('xval_output')
        print '\n\n'
        print 'AUC WITH ALL ELECTRODES: ', self.xval_output[-1].auc
        print 'AUC EXCLUDING STIM-ADJACENT ELECTRODES: ', reduced_xval_output[-1].auc

    def run(self):
        super(ComputeFullClassifier,self).run()
        self.compare_AUCs()


class ComputeEncodingClassifier(ComputeClassifier):

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols','r1.json'))

        hash_md5 = hashlib.md5()

        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)
        for fname in bp_paths:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        fr1_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='FR1')))
        for fname in fr1_event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        catfr1_event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment='catFR1')))
        for fname in catfr1_event_files:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def xval_test_type(self, events):
        event_sessions = events.session
        sessions = np.unique(event_sessions)
        if len(sessions) == 1:
            return 'lolo'
        for sess in sessions:
            sess_events = events[event_sessions == sess]
            if len(sess_events) >= 0.7 * len(events):
                return 'lolo'
        return 'loso'

    @property
    def events(self):
        self._events = self.get_passed_object('FR_events')
        return self._events[self._events.type=='WORD']

    def _normalize_sessions(self,events):
        self.pow_mat = normalize_powers_by_session(self.pow_mat, events)

    def get_pow_mat(self):
        events = self.events
        pow_mat = super(ComputeEncodingClassifier, self).get_pow_mat()
        return pow_mat[self._events.type=='WORD',...]

    def run(self):
        subject = self.pipeline.subject

        events = self.events
        self.pow_mat = self.get_pow_mat()
        self._normalize_sessions(events)

        sample_weights = np.ones(events.shape)
        self.lr_classifier = LogisticRegression(C=self.params.C,
                                                penalty=self.params.penalty_type,
                                                solver='liblinear')

        event_sessions = events.session
        recalls = events.recalled

        if self.xval_test_type(events) == 'loso':
            print 'Performing permutation test'
            self.perm_AUCs = permuted_loso_AUCs(self,event_sessions, recalls)

            print 'Performing leave-one-session-out xval'
            run_loso_xval(event_sessions, recalls,
                                        self.pow_mat, self.lr_classifier,self.xval_output,permuted=False)
        else:
            print 'Performing in-session permutation test'
            self.perm_AUCs = permuted_lolo_AUCs(self,events)

            print 'Performing leave-one-list-out xval'
            run_lolo_xval(events, recalls, self.pow_mat,self.lr_classifier,self.xval_output, permuted=False)

        self.pvalue = np.sum(self.perm_AUCs >= self.xval_output[-1].auc) / float(self.perm_AUCs.size)
        print 'Perm test p-value =', self.pvalue

        print 'thresh =', self.xval_output[-1].jstat_thresh, 'quantile =', self.xval_output[-1].jstat_quantile

        print 'AUC = ',self.xval_output[-1].auc

        # Finally, fitting classifier on all available data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lr_classifier.fit(self.pow_mat, recalls, sample_weights)

        self.pass_objects()

    def pass_objects(self):
        subject=self.pipeline.subject
        classifier_path = self.get_path_to_resource_in_workspace(subject + '-lr_classifier_encoding.pkl')
        self.pass_object('lr_classifier', self.lr_classifier)
        self.pass_object('xval_output', self.xval_output)
        self.pass_object('perm_AUCs', self.perm_AUCs)
        self.pass_object('pvalue', self.pvalue)
        self.pass_object('classifier_path',classifier_path)
        self.pass_object('reduced_pow_mat',self.pow_mat)

        joblib.dump(self.lr_classifier, classifier_path)
        joblib.dump(self.xval_output, self.get_path_to_resource_in_workspace(subject + '-xval_output.pkl'))
        joblib.dump(self.perm_AUCs, self.get_path_to_resource_in_workspace(subject + '-perm_AUCs.pkl'))
        joblib.dump(self.pvalue, self.get_path_to_resource_in_workspace(subject + '-pvalue.pkl'))
        joblib.dump(self.pow_mat,self.get_path_to_resource_in_workspace(subject+ '-reduced_pow_mat.pkl'))

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.args.experiment

        for attr in ['xval_output', 'perm_AUCs', 'pvalue']:
            try:
                self.__setattr__(attr, joblib.load(self.get_path_to_resource_in_workspace(subject + '-%s.pkl' % attr)))
            except IOError:
                self.__setattr__(attr, joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-%s.pkl' % attr)))

        classifier_path = self.get_path_to_resource_in_workspace(subject+'-lr_classifier_encoding.pkl')
        self.pass_object('classifier_path',classifier_path)
        self.lr_classifier = joblib.load(classifier_path)
        self.pass_object('lr_classifier', self.lr_classifier)
        self.pass_object('xval_output', self.xval_output)
        self.pass_object('perm_AUCs', self.perm_AUCs)
        self.pass_object('pvalue', self.pvalue)
