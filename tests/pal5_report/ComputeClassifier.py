import os
import numpy as np
from scipy.stats.mstats import zscore
from sklearn.linear_model import LogisticRegression
from random import shuffle
from sklearn.metrics import roc_auc_score, roc_curve
from ReportTasks.RamTaskMethods import run_lolo_xval, run_loso_xval, permuted_lolo_AUCs, permuted_loso_AUCs, ModelOutput
from sklearn.externals import joblib
import warnings

from ptsa.data.readers  import JsonIndexReader

import hashlib

from ramutils.pipeline import RamTask


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

        # fr1_event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR1')))
        # for fname in fr1_event_files:
        #     with open(fname,'rb') as f: hash_md5.update(f.read())
        #
        # catfr1_event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='catFR1')))
        # for fname in catfr1_event_files:
        #     with open(fname,'rb') as f: hash_md5.update(f.read())
        #
        # fr3_event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR3')))
        # for fname in fr3_event_files:
        #     with open(fname,'rb') as f: hash_md5.update(f.read())
        #
        # catfr3_event_files = sorted(list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='catFR3')))
        # for fname in catfr3_event_files:
        #     with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def get_auc(self, classifier, features, recalls, mask):

        masked_recalls = recalls[mask]
        probs = classifier.predict_proba(features[mask])[:, 1]
        auc = roc_auc_score(masked_recalls, probs)
        return auc

    def run_loso_xval(self, event_sessions, recalls, permuted=False, use_samples_weights=False, events=None):

        probs = np.empty_like(recalls, dtype=np.float)

        sessions = np.unique(event_sessions)

        auc_encoding = np.empty(sessions.shape[0], dtype=np.float)
        auc_retrieval = np.empty(sessions.shape[0], dtype=np.float)
        auc_both = np.empty(sessions.shape[0], dtype=np.float)

        auc_encoding_pal = np.zeros(sessions.shape[0], dtype=np.float)
        auc_retrieval_pal = np.zeros(sessions.shape[0], dtype=np.float)
        auc_both_pal = np.zeros(sessions.shape[0], dtype=np.float)

        auc_encoding_fr = np.zeros(sessions.shape[0], dtype=np.float)
        auc_retrieval_fr = np.zeros(sessions.shape[0], dtype=np.float)
        auc_both_fr = np.zeros(sessions.shape[0], dtype=np.float)

        for sess_idx, sess in enumerate(sessions):
            outsample_classifier = self.create_classifier_obj()
            insample_mask = (event_sessions != sess)

            insample_pow_mat = self.pow_mat[insample_mask]
            insample_recalls = recalls[insample_mask]

            insample_enc_mask = insample_mask & ((events.type == 'WORD'))
            insample_retrieval_mask = insample_mask & (events.type == 'REC_EVENT')

            # n_enc_0 = events[insample_enc_mask & (events.correct == 0)].shape[0]
            # n_enc_1 = events[insample_enc_mask & (events.correct == 1)].shape[0]
            #
            # n_ret_0 = events[insample_retrieval_mask & (events.correct == 0)].shape[0]
            # n_ret_1 = events[insample_retrieval_mask & (events.correct == 1)].shape[0]
            #
            # n_vec = np.array([1.0 / n_enc_0, 1.0 / n_enc_1, 1.0 / n_ret_0, 1.0 / n_ret_1], dtype=np.float)
            # # n_vec /= np.mean(n_vec)
            #
            # n_vec[:2] *= self.params.encoding_samples_weight
            #
            # n_vec /= np.mean(n_vec)
            #
            # # insample_samples_weights = np.ones(n_enc_0 + n_enc_1 + n_ret_0 + n_ret_1, dtype=np.float)
            # insample_samples_weights = np.ones(events.shape[0], dtype=np.float)
            #
            # insample_samples_weights[insample_enc_mask & (events.correct == 0)] = n_vec[0]
            # insample_samples_weights[insample_enc_mask & (events.correct == 1)] = n_vec[1]
            # insample_samples_weights[insample_retrieval_mask & (events.correct == 0)] = n_vec[2]
            # insample_samples_weights[insample_retrieval_mask & (events.correct == 1)] = n_vec[3]
            #
            # insample_samples_weights = insample_samples_weights[insample_mask]

            insample_samples_weights = self.get_sample_weights_vector(evs=events[insample_mask])

            outsample_both_mask = (events.session == sess)

            # % even weights by class balance
            # n_vec = [1/n_enc_pos 1/n_enc_neg 1/n_rec_pos 1/n_rec_neg];
            # mean_tmp = mean(n_vec);
            # n_vec = n_vec/mean_tmp;
            #
            # % add scalign by E
            # n_vec(1:2) = n_vec(1:2)*E;
            # mean_tmp = mean(n_vec);
            # n_vec = n_vec/mean_tmp;


            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if use_samples_weights:
                    outsample_classifier.fit(insample_pow_mat, insample_recalls, insample_samples_weights)
                else:
                    outsample_classifier.fit(insample_pow_mat, insample_recalls)

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

            if events is not None and not permuted:
                outsample_encoding_mask = (events.session == sess) & (events.type == 'WORD')
                outsample_retrieval_mask = (events.session == sess) & (events.type == 'REC_EVENT')
                outsample_both_mask = (events.session == sess)

                outsample_encoding_pal_mask = (events.session == sess) & (events.type == 'WORD') & (
                    events.exp_name == 'PAL1')
                outsample_retrieval_pal_mask = (events.session == sess) & (events.type == 'REC_EVENT') & (
                    events.exp_name == 'PAL1')
                outsample_both_pal_mask = (events.session == sess) & (events.exp_name == 'PAL1')

                outsample_encoding_fr_mask = (events.session == sess) & (events.type == 'WORD') & (
                    events.exp_name == 'FR1')
                outsample_retrieval_fr_mask = (events.session == sess) & (events.type == 'REC_EVENT') & (
                    events.exp_name == 'FR1')
                outsample_both_fr_mask = (events.session == sess) & (events.exp_name == 'FR1')

                # print 'num outsample_encoding_pal = ', np.sum(outsample_encoding_pal_mask.astype(np.int))
                # print 'num outsample_retrieval_pal_mask = ', np.sum(outsample_retrieval_pal_mask.astype(np.int))
                # print 'num outsample_both_pal_mask = ', np.sum(outsample_both_pal_mask.astype(np.int))
                #

                auc_encoding[sess_idx] = self.get_auc(
                    classifier=outsample_classifier, features=self.pow_mat, recalls=recalls,
                    mask=outsample_encoding_mask)

                auc_retrieval[sess_idx] = self.get_auc(
                    classifier=outsample_classifier, features=self.pow_mat, recalls=recalls,
                    mask=outsample_retrieval_mask)

                auc_both[sess_idx] = self.get_auc(
                    classifier=outsample_classifier, features=self.pow_mat, recalls=recalls, mask=outsample_both_mask)

                # testing PAL1 only here
                if np.sum(outsample_encoding_pal_mask.astype(np.int)) != 0:
                    auc_encoding_pal[sess_idx] = self.get_auc(
                        classifier=outsample_classifier, features=self.pow_mat, recalls=recalls,
                        mask=outsample_encoding_pal_mask)

                    auc_retrieval_pal[sess_idx] = self.get_auc(
                        classifier=outsample_classifier, features=self.pow_mat, recalls=recalls,
                        mask=outsample_retrieval_pal_mask)

                    auc_both_pal[sess_idx] = self.get_auc(
                        classifier=outsample_classifier, features=self.pow_mat, recalls=recalls,
                        mask=outsample_both_pal_mask)

                # testing FR1 only here
                if np.sum(outsample_encoding_fr_mask.astype(np.int)) != 0:
                    auc_encoding_fr[sess_idx] = self.get_auc(
                        classifier=outsample_classifier, features=self.pow_mat, recalls=recalls,
                        mask=outsample_encoding_fr_mask)

                    auc_retrieval_fr[sess_idx] = self.get_auc(
                        classifier=outsample_classifier, features=self.pow_mat, recalls=recalls,
                        mask=outsample_retrieval_fr_mask)

                    auc_both_fr[sess_idx] = self.get_auc(
                        classifier=outsample_classifier, features=self.pow_mat, recalls=recalls,
                        mask=outsample_both_fr_mask)

        if not permuted:
            self.xval_output[-1] = ModelOutput(recalls, probs)
            self.xval_output[-1].compute_roc()
            self.xval_output[-1].compute_tercile_stats()

            print '----------------TESTING PAL1 AND FR1----------------------- '
            print 'auc_encoding=', auc_encoding, np.mean(auc_encoding)
            print 'auc_retrieval=', auc_retrieval, np.mean(auc_retrieval)
            print 'auc_both=', auc_both, np.mean(auc_both)

            print '\n\n'

            print '----------------TESTING PAL1 ONLY----------------------- '
            print 'auc_encoding_pal=', auc_encoding_pal, np.mean(auc_encoding_pal[auc_encoding_pal > 0.0])
            print 'auc_retrieval_pal=', auc_retrieval_pal, np.mean(auc_retrieval_pal[auc_encoding_pal > 0.0])
            print 'auc_both_pal=', auc_both_pal, np.mean(auc_both_pal[auc_encoding_pal > 0.0])

            print '\n\n'

            print '----------------TESTING FR1 ONLY----------------------- '
            print 'auc_encoding_fr=', auc_encoding_fr, np.mean(auc_encoding_fr[auc_encoding_fr > 0.0])
            print 'auc_retrieval_fr=', auc_retrieval_fr, np.mean(auc_retrieval_fr[auc_encoding_fr > 0.0])
            print 'auc_both_fr=', auc_both_fr, np.mean(auc_both_fr[auc_encoding_fr > 0.0])

            print '\n\n'

            # for sess_key in sorted(self.xval_output.keys()):
            #     xval_out = self.xval_output[sess_key]
            #     print '-----------sess %s' % str(sess_key)
            #     print 'AUC: ', xval_out.auc
            #     print 'median classifier = ',xval_out.jstat_thresh

        self.pass_object('auc_encoding' + self.suffix, auc_encoding)
        self.pass_object('auc_retrieval' + self.suffix, auc_retrieval)
        self.pass_object('auc_both' + self.suffix, auc_both)

        return probs

    def permuted_loso_AUCs(self, event_sessions, recalls, use_samples_weights=False, events=None):
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
                                           use_samples_weights=use_samples_weights, events=events)
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
        try:
            pow_mat = pow_mat.reshape((len(pow_mat), -1, len(self.params.freqs)))[:, to_include, :]
            return pow_mat.reshape((len(pow_mat), -1))
        except IndexError:
            return pow_mat

    def pass_objects(self):
        subject = self.pipeline.subject
        self.pass_object('lr_classifier',self.lr_classifier)
        self.pass_object('xval_output' , self.xval_output)
        self.pass_object('perm_AUCs' , self.perm_AUCs)
        self.pass_object('pvalue' , self.pvalue)

        classifier_path = self.get_path_to_resource_in_workspace(subject + '-lr_classifier' + self.suffix + '.pkl')
        joblib.dump(self.lr_classifier, classifier_path)
        # joblib.dump(self.lr_classifier, self.get_path_to_resource_in_workspace(subject + '-lr_classifier.pkl'))
        joblib.dump(self.xval_output,
                    self.get_path_to_resource_in_workspace(subject + '-xval_output' + self.suffix + '.pkl'))
        joblib.dump(self.perm_AUCs,
                    self.get_path_to_resource_in_workspace(subject + '-perm_AUCs' + self.suffix + '.pkl'))
        joblib.dump(self.pvalue, self.get_path_to_resource_in_workspace(subject + '-pvalue' + self.suffix + '.pkl'))


    def create_classifier_obj(self):
        return LogisticRegression(C=self.params.C, penalty=self.params.penalty_type, class_weight='balanced',
                                  solver='newton-cg')

    def get_sample_weights_vector(self, evs):
        """
        Computes vector of sample weihghts taking int account number fo 0'1 1's ,
        whether the sample is retrieval or encoding. Or whether the sample is from PAL or from FR experiment.
        The weighting should be desribed in detail in the design doc
        :param evs: events
        :return: {ndarray} vector of sample weights
        """
        # evs = evs.view(np.recarray)
        enc_mask = (evs.type == 'WORD')
        retrieval_mask = (evs.type == 'REC_EVENT')

        pal_mask = (evs.exp_name == 'PAL1')
        fr_mask = ~pal_mask

        pal_n_enc_0 = evs[pal_mask & enc_mask & (evs.correct == 0)].shape[0]
        pal_n_enc_1 = evs[pal_mask & enc_mask & (evs.correct == 1)].shape[0]

        pal_n_ret_0 = evs[pal_mask & retrieval_mask & (evs.correct == 0)].shape[0]
        pal_n_ret_1 = evs[pal_mask & retrieval_mask & (evs.correct == 1)].shape[0]

        fr_n_enc_0 = evs[fr_mask & enc_mask & (evs.correct == 0)].shape[0]
        fr_n_enc_1 = evs[fr_mask & enc_mask & (evs.correct == 1)].shape[0]

        fr_n_ret_0 = evs[fr_mask & retrieval_mask & (evs.correct == 0)].shape[0]
        fr_n_ret_1 = evs[fr_mask & retrieval_mask & (evs.correct == 1)].shape[0]

        ev_count_list = [pal_n_enc_0, pal_n_enc_1, pal_n_ret_0, pal_n_ret_1, fr_n_enc_0, fr_n_enc_1, fr_n_ret_0,
                         fr_n_ret_1]

        n_vec = np.array([0.0] * 8, dtype=np.float)

        for i, ev_count in enumerate(ev_count_list):
            n_vec[i] = 1. / ev_count if ev_count else 0.0

        n_vec /= np.mean(n_vec)

        # scaling PAL1 task
        n_vec[0:4] *= self.params.pal_samples_weight
        n_vec /= np.mean(n_vec)

        # scaling encoding
        n_vec[[0, 1, 4, 5]] *= self.params.encoding_samples_weight
        n_vec /= np.mean(n_vec)

        samples_weights = np.ones(evs.shape[0], dtype=np.float)

        samples_weights[pal_mask & enc_mask & (evs.correct == 0)] = n_vec[0]
        samples_weights[pal_mask & enc_mask & (evs.correct == 1)] = n_vec[1]
        samples_weights[pal_mask & retrieval_mask & (evs.correct == 0)] = n_vec[2]
        samples_weights[pal_mask & retrieval_mask & (evs.correct == 1)] = n_vec[3]

        samples_weights[fr_mask & enc_mask & (evs.correct == 0)] = n_vec[4]
        samples_weights[fr_mask & enc_mask & (evs.correct == 1)] = n_vec[5]
        samples_weights[fr_mask & retrieval_mask & (evs.correct == 0)] = n_vec[6]
        samples_weights[fr_mask & retrieval_mask & (evs.correct == 1)] = n_vec[7]

        return samples_weights

    def run_classifier_pipeline(self, evs):

        encoding_mask = (evs.type == 'WORD')

        # pow_mat = self.filter_pow_mat()
        pow_mat_copy = np.copy(self.filter_pow_mat())

        self.pow_mat = self.filter_pow_mat()
        self.pow_mat[encoding_mask] = normalize_sessions(self.pow_mat[encoding_mask], evs[encoding_mask])
        self.pow_mat[~encoding_mask] = normalize_sessions(self.pow_mat[~encoding_mask], evs[~encoding_mask])

        # computing z-scoring vectors

        mean_dict, std_dict = compute_z_scoring_vecs(pow_mat_copy[~encoding_mask], evs[~encoding_mask])

        self.pass_object('features_mean_dict', mean_dict)
        self.pass_object('features_std_dict', std_dict)

        print

        self.lr_classifier = LogisticRegression(C=self.params.C, penalty=self.params.penalty_type, class_weight='balanced',
                                                solver='newton-cg')

        sessions_array = evs.session

        recalls = evs.correct

        sessions = np.unique(sessions_array)
        if len(sessions) > 1:
            print 'Performing permutation test'
            self.perm_AUCs = self.permuted_loso_AUCs(sessions_array, recalls, use_samples_weights=True, events=evs)
            print 'Performing leave-one-session-out xval'
            self.run_loso_xval(sessions_array, recalls, permuted=False, use_samples_weights=True, events=evs)


        else:
            raise RuntimeError("Training of the combined PAL1 & FR1 classifier requires at least two sessions")

        print 'CROSS VALIDATION AUC =', self.xval_output[-1].auc

        self.pvalue = np.nansum(self.perm_AUCs >= self.xval_output[-1].auc) / float(
            self.perm_AUCs[~np.isnan(self.perm_AUCs)].size)
        print 'Perm test p-value =', self.pvalue

        print 'thresh =', self.xval_output[-1].jstat_thresh, 'quantile =', self.xval_output[-1].jstat_quantile

        # Finally, fitting classifier on all available data
        samples_weights = self.get_sample_weights_vector(evs=evs)
        self.lr_classifier.fit(self.pow_mat, recalls, samples_weights)

        # FYI - in-sample AUC
        recall_prob_array = self.lr_classifier.predict_proba(self.pow_mat)[:, 1]
        insample_auc = roc_auc_score(recalls, recall_prob_array)
        print 'in-sample AUC=', insample_auc

        # print 'training retrieval_clasifiers = ', recall_prob_array[evs.type == 'REC_EVENT']
        self.pass_object('rec_pow_mat', self.pow_mat[evs.type == 'REC_EVENT'])

        self.pass_objects()

    def run(self):

        evs = self.get_passed_object('combined_evs')
        self.run_classifier_pipeline(evs)

    def restore(self):
        subject = self.pipeline.subject

        classifier_path = self.get_path_to_resource_in_workspace(subject + '-lr_classifier{}.pkl'.format(self.suffix))
        self.lr_classifier = joblib.load(classifier_path)
        # self.lr_classifier = joblib.load(self.get_path_to_resource_in_workspace(subject + '-lr_classifier.pkl'))
        self.xval_output = joblib.load(self.get_path_to_resource_in_workspace(subject + '-xval_output{}.pkl'.format(self.suffix)))
        self.perm_AUCs = joblib.load(self.get_path_to_resource_in_workspace(subject + '-perm_AUCs{}.pkl'.format(self.suffix)))
        self.pvalue = joblib.load(self.get_path_to_resource_in_workspace(subject + '-pvalue{}.pkl'.format(self.suffix)))

        self.pass_object('classifier_path', classifier_path)
        self.pass_object('lr_classifier', self.lr_classifier)
        self.pass_object('xval_output', self.xval_output)
        self.pass_object('perm_AUCs', self.perm_AUCs)
        self.pass_object('pvalue', self.pvalue)


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


class ComputePAL1Classifier(ComputeClassifier):
    # def restore(self):
    #     subject = self.pipeline.subject
    #     full_classifier_path = self.get_path_to_resource_in_workspace(subject + '-xval_output_all_electrodes.pkl')
    #     self.xval_output = joblib.load(full_classifier_path)
    #     self.compare_AUCs()
    #     self.pass_object('full_classifier_path', full_classifier_path)
    #     self.pass_object('xval_output_all_electrodes', self.xval_output)
    #
    # def pass_objects(self):
    #     subject = self.pipeline.subject
    #     classifier_path = self.get_path_to_resource_in_workspace(subject + 'lr_classifier_full.pkl')
    #     joblib.dump(self.lr_classifier, classifier_path)
    #     joblib.dump(self.xval_output,
    #                 self.get_path_to_resource_in_workspace(subject + '-xval_output_all_electrodes.pkl'))
    #     self.pass_object('classifier_path', classifier_path)
    #     self.pass_object('xval_output_all_electrodes', self.xval_output)

    def __init__(self,*args,**kwargs):
        super(ComputePAL1Classifier, self).__init__(*args,**kwargs)
        self.suffix = '_pal'


    def compare_AUCs(self):
        reduced_xval_output = self.get_passed_object('xval_output')
        print '\n\n'
        print 'AUC WITH ALL ELECTRODES: ', self.xval_output[-1].auc
        print 'AUC EXCLUDING STIM-ADJACENT ELECTRODES: ', reduced_xval_output[-1].auc

    def filter_pow_mat(self):
        pow_mat = super(ComputePAL1Classifier, self).filter_pow_mat()
        evs = self.get_passed_object('combined_evs')

        pow_mat = pow_mat[evs.exp_name == 'PAL1']

        return pow_mat

    def run(self):
        evs = self.get_passed_object('combined_evs')
        evs = evs[evs.exp_name == 'PAL1']

        print '\n\n ---------------- PAL1 CLASSIFIER ONLY------------------\n\n'

        sessions_array = evs.session

        if len(np.unique(sessions_array)) < 2:
            warnings.warn('SKIPPING PAL1-only classifier because it needs more than one session of PAL1 ',
                          RuntimeWarning)
            return

        super(ComputePAL1Classifier, self).run_classifier_pipeline(evs)

