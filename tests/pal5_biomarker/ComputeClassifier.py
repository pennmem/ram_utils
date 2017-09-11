from RamPipeline import *

import numpy as np
from scipy.stats.mstats import zscore
from sklearn.linear_model import LogisticRegression
from random import shuffle
from sklearn.metrics import roc_auc_score, roc_curve
from ReportTasks.RamTaskMethods import run_lolo_xval, run_loso_xval, permuted_lolo_AUCs, permuted_loso_AUCs, ModelOutput
from sklearn.externals import joblib
import warnings

from ptsa.data.readers.IndexReader import JsonIndexReader

import hashlib


def normalize_sessions(pow_mat, events):
    sessions = np.unique(events.session)
    for sess in sessions:
        sess_event_mask = (events.session == sess)
        pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask], axis=0, ddof=1)
    return pow_mat

def compute_z_scoring_vecs(pow_mat, events):

    mean_dict = {}
    std_dict = {}
    sessions = np.unique(events.session)
    for sess in sessions:
        sess_event_mask = (events.session == sess)

        m = np.mean(pow_mat[sess_event_mask], axis=0)
        s = np.std(pow_mat[sess_event_mask], axis=0, ddof=1)

        mean_dict[sess] = m
        std_dict[sess] = s


    return mean_dict, std_dict
        # pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask], axis=0, ddof=1)

    # self.m = np.mean(mp_rs, axis=0)
    # self.s = np.std(mp_rs, axis=0, ddof=1)



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
        self.classifier = None

    def compute_roc(self):
        try:
            self.auc = roc_auc_score(self.true_labels, self.probs)
        except ValueError:
            return
        self.fpr, self.tpr, self.thresholds = roc_curve(self.true_labels, self.probs)
        # idx = np.argmax(self.tpr-self.fpr)
        # self.jstat_thresh = self.thresholds[idx]
        # self.jstat_quantile = np.sum(self.probs <= self.jstat_thresh) / float(self.probs.size)
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
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.lr_classifier = None
        self.xval_output = dict()  # ModelOutput per session; xval_output[-1] is across all sessions
        self.perm_AUCs = None
        self.pvalue = None
        self.suffix = ''

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)
        for fname in bp_paths:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def get_auc(self, classifier, features, recalls, mask):

        masked_recalls = recalls[mask]
        probs = classifier.predict_proba(features[mask])[:, 1]
        auc = roc_auc_score(masked_recalls, probs)
        return auc

    def run_loso_xval(self, event_sessions, recalls, permuted=False, samples_weights=None, events=None):

        # outsample_classifier = self.create_classifier_obj()

        probs = np.empty_like(recalls, dtype=np.float)

        sessions = np.unique(event_sessions)

        auc_encoding = np.empty(sessions.shape[0], dtype=np.float)
        auc_retrieval = np.empty(sessions.shape[0], dtype=np.float)
        auc_both = np.empty(sessions.shape[0], dtype=np.float)

        for sess_idx, sess in enumerate(sessions):
            outsample_classifier = self.create_classifier_obj()
            insample_mask = (event_sessions != sess)



            insample_pow_mat = self.pow_mat[insample_mask]
            insample_recalls = recalls[insample_mask]
            insample_samples_weights = samples_weights[insample_mask]

            insample_enc_mask = insample_mask & ((events.type == 'STUDY_PAIR'))
            insample_retrieval_mask = insample_mask & (events.type == 'REC_EVENT')


            n_enc_0 = events[insample_enc_mask & (events.correct == 0)].shape[0]
            n_enc_1 = events[insample_enc_mask & (events.correct == 1)].shape[0]

            n_ret_0 = events[insample_retrieval_mask & (events.correct == 0)].shape[0]
            n_ret_1 = events[insample_retrieval_mask & (events.correct == 1)].shape[0]

            n_vec = np.array([1.0 / n_enc_0, 1.0 / n_enc_1, 1.0 / n_ret_0, 1.0 / n_ret_1], dtype=np.float)
            n_vec /= np.mean(n_vec)

            n_vec[:2] *= self.params.encoding_samples_weight

            n_vec /= np.mean(n_vec)

            # insample_samples_weights = np.ones(n_enc_0 + n_enc_1 + n_ret_0 + n_ret_1, dtype=np.float)
            insample_samples_weights = np.ones(events.shape[0], dtype=np.float)

            insample_samples_weights[insample_enc_mask & (events.correct == 0)] = n_vec[0]
            insample_samples_weights[insample_enc_mask & (events.correct == 1)] = n_vec[1]
            insample_samples_weights[insample_retrieval_mask & (events.correct == 0)] = n_vec[2]
            insample_samples_weights[insample_retrieval_mask & (events.correct == 1)] = n_vec[3]

            insample_samples_weights = insample_samples_weights[insample_mask]

            outsample_both_mask = (events.session == sess)


            # TODO ORIGINAL CODE
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if samples_weights is not None:
                    outsample_classifier.fit(insample_pow_mat, insample_recalls, insample_samples_weights)
                else:
                    outsample_classifier.fit(insample_pow_mat, insample_recalls)

            # # todo - code with no weighting
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     outsample_classifier.fit(insample_pow_mat, insample_recalls)

            outsample_mask = ~insample_mask
            outsample_pow_mat = self.pow_mat[outsample_mask]
            outsample_recalls = recalls[outsample_mask]

            outsample_probs = outsample_classifier.predict_proba(outsample_pow_mat)[:, 1]
            if not permuted:
                self.xval_output[sess] = ModelOutput(outsample_recalls, outsample_probs)
                self.xval_output[sess].compute_roc()
                self.xval_output[sess].compute_tercile_stats()
                self.xval_output[sess].classifier = outsample_classifier

            probs[outsample_mask] = outsample_probs


            if events is not None:
                outsample_encoding_mask = (events.session == sess) & ((events.type == 'STUDY_PAIR')|(events.type == 'PRACTICE_PAIR'))
                outsample_retrieval_mask = (events.session == sess) & ((events.type == 'REC_EVENT'))



                outsample_both_mask = (events.session == sess)

                auc_encoding[sess_idx] = self.get_auc(
                    classifier=outsample_classifier, features=self.pow_mat, recalls=recalls, mask=outsample_encoding_mask)

                auc_retrieval[sess_idx] = self.get_auc(
                    classifier=outsample_classifier, features=self.pow_mat, recalls=recalls,
                    mask=outsample_retrieval_mask)

                auc_both[sess_idx] = self.get_auc(
                    classifier=outsample_classifier, features=self.pow_mat, recalls=recalls, mask=outsample_both_mask)



        if not permuted:
            self.xval_output[-1] = ModelOutput(recalls, probs)
            self.xval_output[-1].compute_roc()
            self.xval_output[-1].compute_tercile_stats()


            print 'auc_encoding=', auc_encoding, np.mean(auc_encoding)
            print 'auc_retrieval=', auc_retrieval, np.mean(auc_retrieval)
            print 'auc_both=', auc_both, np.mean(auc_both)

        joblib.dump({'auc_encoding'+self.suffix:auc_encoding,
                     'auc_retrieval'+self.suffix:auc_retrieval,
                     'auc_both'+self.suffix:auc_both},self.get_path_to_resource_in_workspace('aucs.pkl'))
        self.pass_object('auc_encoding'+self.suffix, auc_encoding)
        self.pass_object('auc_retrieval'+self.suffix, auc_retrieval)
        self.pass_object('auc_both'+self.suffix, auc_both)


        return probs

    def permuted_loso_AUCs(self, event_sessions, recalls, samples_weights=None, events=None):
        n_perm = self.params.n_perm
        permuted_recalls = np.array(recalls)
        AUCs = np.empty(shape=n_perm, dtype=np.float)
        for i in xrange(n_perm):
            try:
                for sess in event_sessions:
                    sel = (event_sessions == sess)
                    sess_permuted_recalls = permuted_recalls[sel]
                    shuffle(sess_permuted_recalls)
                    permuted_recalls[sel] = sess_permuted_recalls
                probs = self.run_loso_xval(event_sessions, permuted_recalls, permuted=True,
                                           samples_weights=samples_weights, events=events)
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
                    self.lr_classifier.fit(insample_pow_mat, insample_recalls, insample_samples_weights)
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

    def permuted_lolo_AUCs(self, sess, event_lists, recalls, samples_weights=None):
        n_perm = self.params.n_perm
        permuted_recalls = np.array(recalls)
        AUCs = np.empty(shape=n_perm, dtype=np.float)
        for i in xrange(n_perm):
            for lst in event_lists:
                sel = (event_lists == lst)
                list_permuted_recalls = permuted_recalls[sel]
                shuffle(list_permuted_recalls)
                permuted_recalls[sel] = list_permuted_recalls
            probs = self.run_lolo_xval(sess, event_lists, permuted_recalls, permuted=True,
                                       samples_weights=samples_weights)
            AUCs[i] = roc_auc_score(recalls, probs)
            print 'AUC =', AUCs[i]
        return AUCs

    def filter_pow_mat(self):
        """
        This function filters power matrix to exclude certain bipolar pairs - here the ones that "touch" stimulated
        electrodes
        :return: None
        """
        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        reduced_pairs = self.get_passed_object('reduced_pairs')
        to_include = np.array([bp in reduced_pairs for bp in bipolar_pairs])
        pow_mat = self.get_passed_object('pow_mat')
        pow_mat = pow_mat.reshape((len(pow_mat), len(bipolar_pairs), -1))[:, to_include, :]
        return pow_mat.reshape((len(pow_mat), -1))

    def pass_objects(self):
        subject = self.pipeline.subject
        self.pass_object('lr_classifier', self.lr_classifier)
        self.pass_object('xval_output', self.xval_output)
        self.pass_object('perm_AUCs', self.perm_AUCs)
        self.pass_object('pvalue', self.pvalue)

        classifier_path = self.get_path_to_resource_in_workspace(subject + '-lr_classifier.pkl')
        joblib.dump(self.lr_classifier, classifier_path)
        # joblib.dump(self.lr_classifier, self.get_path_to_resource_in_workspace(subject + '-lr_classifier.pkl'))
        joblib.dump(self.xval_output, self.get_path_to_resource_in_workspace(subject + '-xval_output.pkl'))
        joblib.dump(self.perm_AUCs, self.get_path_to_resource_in_workspace(subject + '-perm_AUCs.pkl'))
        joblib.dump(self.pvalue, self.get_path_to_resource_in_workspace(subject + '-pvalue.pkl'))

        self.pass_object('classifier_path', classifier_path)


    def create_classifier_obj(self):
        return LogisticRegression(C=self.params.C, penalty=self.params.penalty_type, class_weight='auto',fit_intercept=False,
                                                        solver='newton-cg')


    def run(self):

        events = self.get_passed_object('PAL1_events')
        # self.get_pow_mat() is essential - it does the filtering on the

        encoding_mask = (events.type == 'STUDY_PAIR') | (events.type == 'PRACTICE_PAIR')

        # pow_mat = self.filter_pow_mat()
        pow_mat_copy = np.copy(self.filter_pow_mat())

        self.pow_mat = self.filter_pow_mat()
        self.pow_mat[encoding_mask] = normalize_sessions(self.pow_mat[encoding_mask], events[encoding_mask])
        self.pow_mat[~encoding_mask] = normalize_sessions(self.pow_mat[~encoding_mask], events[~encoding_mask])

        # computing z-scoring vectors

        mean_dict, std_dict = compute_z_scoring_vecs(pow_mat_copy[~encoding_mask], events[~encoding_mask])


        self.pass_object('features_mean_dict', mean_dict)
        self.pass_object('features_std_dict', std_dict)
        joblib.dump(mean_dict,self.get_path_to_resource_in_workspace('features_mean_dict.pkl'),)
        joblib.dump(std_dict,self.get_path_to_resource_in_workspace('features_std_dict.pkl'),)


        self.lr_classifier = LogisticRegression(C=self.params.C, penalty=self.params.penalty_type, class_weight='auto',fit_intercept=False,
                                                solver='newton-cg')

        event_sessions = events.session

        recalls = events.correct

        samples_weights = np.ones(events.shape[0], dtype=np.float)

        # samples_weights[~(events.type=='WORD')] = self.params.retrieval_samples_weight

        samples_weights[
            (events.type == 'STUDY_PAIR') | (events.type == 'PRACTICE_PAIR')] = self.params.encoding_samples_weight

        sessions = np.unique(event_sessions)
        if len(sessions) > 1:
            print 'Performing permutation test'
            self.perm_AUCs = self.permuted_loso_AUCs(event_sessions, recalls, samples_weights, events=events)

            print 'Performing leave-one-session-out xval'
            self.run_loso_xval(event_sessions, recalls, permuted=False, samples_weights=samples_weights, events=events)
        else:
            sess = sessions[0]
            event_lists = events.list

            print 'Performing in-session permutation test'
            self.perm_AUCs = self.permuted_lolo_AUCs(sess, event_lists, recalls, samples_weights=samples_weights)

            print 'Performing leave-one-list-out xval'
            self.run_lolo_xval(sess, event_lists, recalls, permuted=False, samples_weights=samples_weights)

        print 'CROSS VALIDATION AUC =', self.xval_output[-1].auc

        self.pvalue = np.nansum(self.perm_AUCs >= self.xval_output[-1].auc) / float(
            self.perm_AUCs[~np.isnan(self.perm_AUCs)].size)
        print 'Perm test p-value =', self.pvalue

        print 'thresh =', self.xval_output[-1].jstat_thresh, 'quantile =', self.xval_output[-1].jstat_quantile

        # Finally, fitting classifier on all available data
        self.lr_classifier.fit(self.pow_mat, recalls, samples_weights)

        # FYI - in-sample AUC
        recall_prob_array = self.lr_classifier.predict_proba(self.pow_mat)[:, 1]
        insample_auc = roc_auc_score(recalls, recall_prob_array)
        print 'in-sample AUC=', insample_auc

        print 'training retrieval_clasifiers = ', recall_prob_array[events.type=='REC_EVENT']
        self.pass_object('rec_pow_mat',self.pow_mat[events.type=='REC_EVENT'])

        self.pass_objects()

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

        aucs_dict = joblib.load(self.get_path_to_resource_in_workspace('aucs.pkl'))
        for k in aucs_dict:
            self.pass_object(k,aucs_dict[k])

        mean_dict=joblib.load(self.get_path_to_resource_in_workspace('features_mean_dict.pkl'))
        std_dict = joblib.load(self.get_path_to_resource_in_workspace('features_std_dict.pkl'))
        self.pass_object('features_mean_dict', mean_dict)
        self.pass_object('features_std_dict', std_dict)



class ComputeFullClassifier(ComputeClassifier):
    def filter_pow_mat(self):
        """
        This function filters power matrix to exclude certain bipolar pairs.However,
        this implementation does not do any filtering

        :return: None
        """

        return self.get_passed_object('pow_mat')

    def restore(self):
        subject = self.pipeline.subject
        full_classifier_path = self.get_path_to_resource_in_workspace(subject + '-xval_output_all_electrodes.pkl')
        self.xval_output = joblib.load(full_classifier_path)
        self.compare_AUCs()
        self.pass_object('full_classifier_path', full_classifier_path)
        self.pass_object('xval_output_all_electrodes', self.xval_output)

    def pass_objects(self):
        subject = self.pipeline.subject
        classifier_path = self.get_path_to_resource_in_workspace(subject + 'lr_classifier_full.pkl')
        joblib.dump(self.lr_classifier, classifier_path)
        joblib.dump(self.xval_output,
                    self.get_path_to_resource_in_workspace(subject + '-xval_output_all_electrodes.pkl'))
        self.pass_object('full_classifier_path', classifier_path)
        self.pass_object('xval_output_all_electrodes', self.xval_output)

    def compare_AUCs(self):
        reduced_xval_output = self.get_passed_object('xval_output')
        print '\n\n'
        print 'AUC WITH ALL ELECTRODES: ', self.xval_output[-1].auc
        print 'AUC EXCLUDING STIM-ADJACENT ELECTRODES: ', reduced_xval_output[-1].auc

    def run(self):
        self.suffix = '_full'
        super(ComputeFullClassifier, self).run()
        self.compare_AUCs()


