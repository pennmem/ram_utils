from RamPipeline import *

import numpy as np
from scipy.stats.mstats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from random import shuffle
from sklearn.externals import joblib
from ptsa.data.readers.IndexReader import JsonIndexReader

import hashlib
import warnings

def normalize_sessions(pow_mat, events):
    sessions = np.unique(events.session)
    for sess in sessions:
        sess_event_mask = (events.session == sess)
        pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask], axis=0, ddof=1)
    return pow_mat


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


    def run_loso_xval(self, event_sessions, recalls, permuted=False,samples_weights=None, events=None):
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
            insample_samples_weights = samples_weights[insample_mask]


            insample_enc_mask = insample_mask & (events.type == 'WORD')
            insample_retrieval_mask = insample_mask & ((events.type == 'REC_BASE') | (events.type == 'REC_WORD'))

            n_enc_0 = events[insample_enc_mask & (events.recalled == 0)].shape[0]
            n_enc_1 = events[insample_enc_mask & (events.recalled == 1)].shape[0]

            n_ret_0 = events[insample_retrieval_mask & (events.type == 'REC_BASE')].shape[0]
            n_ret_1 = events[insample_retrieval_mask & (events.type == 'REC_WORD')].shape[0]

            n_vec = np.array([1.0/n_enc_0, 1.0/n_enc_1, 1.0/n_ret_0, 1.0/n_ret_1 ], dtype=np.float)
            n_vec /= np.mean(n_vec)

            n_vec[:2] *= self.params.encoding_samples_weight

            n_vec /= np.mean(n_vec)

            # insample_samples_weights = np.ones(n_enc_0 + n_enc_1 + n_ret_0 + n_ret_1, dtype=np.float)
            insample_samples_weights = np.ones(events.shape[0], dtype=np.float)

            insample_samples_weights [insample_enc_mask & (events.recalled == 0)] = n_vec[0]
            insample_samples_weights [insample_enc_mask & (events.recalled == 1)] = n_vec[1]
            insample_samples_weights [insample_retrieval_mask & (events.type == 'REC_BASE')] = n_vec[2]
            insample_samples_weights [insample_retrieval_mask & (events.type == 'REC_WORD')] = n_vec[3]

            insample_samples_weights = insample_samples_weights[insample_mask]


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
                if samples_weights is not None:
                    self.lr_classifier.fit(insample_pow_mat, insample_recalls,insample_samples_weights)
                else:
                    self.lr_classifier.fit(insample_pow_mat, insample_recalls)


            outsample_mask = ~insample_mask
            outsample_pow_mat = self.pow_mat[outsample_mask]
            outsample_recalls = recalls[outsample_mask]

            outsample_probs = self.lr_classifier.predict_proba(outsample_pow_mat)[:, 1]
            if not permuted:
                self.xval_output[sess] = ModelOutput(outsample_recalls, outsample_probs)
                self.xval_output[sess].compute_roc()
                self.xval_output[sess].compute_tercile_stats()
            probs[outsample_mask] = outsample_probs


            # import tables
            #
            # h5file = tables.open_file('%s_fold_%d.h5'%(self.pipeline.subject, sess), mode='w', title="Test Array")
            # root = h5file.root
            # h5file.create_array(root, "insample_recalls", insample_recalls)
            # h5file.create_array(root, "insample_pow_mat", insample_pow_mat)
            # h5file.create_array(root, "insample_samples_weights", insample_samples_weights)
            # h5file.create_array(root, "outsample_recalls", outsample_recalls)
            # h5file.create_array(root, "outsample_pow_mat", outsample_pow_mat)
            # h5file.create_array(root, "outsample_probs", outsample_probs)
            # h5file.create_array(root, "lr_classifier_coef", self.lr_classifier.coef_)
            # h5file.create_array(root, "lr_classifier_intercept", self.lr_classifier.intercept_)
            #
            # h5file.close()
            #


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


                # outsample_encoding_recalls = recalls[outsample_encoding_mask]
                # outsample_retrieval_recalls = recalls[outsample_retrieval_mask]
                #
                # outsample_probs_encoding = self.lr_classifier.predict_proba(self.pow_mat[outsample_encoding_mask])[:, 1]
                # outsample_probs_retrieval = self.lr_classifier.predict_proba(self.pow_mat[outsample_retrieval_mask])[:, 1]
                #
                # outsample_encoding_auc = roc_auc_score(outsample_encoding_recalls, outsample_probs_encoding)
                # outsample_retrieval_auc = roc_auc_score(outsample_retrieval_recalls, outsample_probs_retrieval)
                #
                #
                # , outsample_encoding_auc
                # print 'outsample_retrieval_auc= ', outsample_retrieval_auc


        if not permuted:
            self.xval_output[-1] = ModelOutput(recalls[events.type=='WORD'], probs[events.type=='WORD'])
            self.xval_output[-1].compute_roc()
            self.xval_output[-1].compute_tercile_stats()

            # outsample_encoding_mask = (events.type == 'WORD')
            # outsample_retrieval_mask =((events.type == 'REC_BASE') | (events.type == 'REC_WORD'))
            #
            # outsample_encoding_recalls = recalls[outsample_encoding_mask]
            # outsample_retrieval_recalls = recalls[outsample_retrieval_mask]
            #
            # outsample_probs_encoding = self.lr_classifier.predict_proba(self.pow_mat[outsample_encoding_mask])[:, 1]
            # outsample_probs_retrieval = self.lr_classifier.predict_proba(self.pow_mat[outsample_retrieval_mask])[:, 1]
            #
            # outsample_encoding_auc = roc_auc_score(outsample_encoding_recalls, outsample_probs_encoding)
            # outsample_retrieval_auc = roc_auc_score(outsample_retrieval_recalls, outsample_probs_retrieval)
            #
            # print 'TOTAL outsample_encoding_auc =', outsample_encoding_auc
            # print 'TOTAL outsample_retrieval_auc= ', outsample_retrieval_auc

            print 'auc_encoding=',auc_encoding, np.mean(auc_encoding)
            print 'auc_retrieval=',auc_retrieval, np.mean(auc_retrieval)
            print 'auc_both=',auc_both, np.mean(auc_both)

            print 'Cross-Validated Encoding AUC = ', roc_auc_score(events[events.type=='WORD'].recalled==1,y_score=encoding_probs)


        return probs

    def permuted_loso_AUCs(self, event_sessions, recalls, samples_weights=None,events=None):
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
                probs = self.run_loso_xval(event_sessions, permuted_recalls, permuted=True,samples_weights=samples_weights,events=events)
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

    def get_pow_mat(self):
        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        reduced_pairs = self.get_passed_object('reduced_pairs')
        to_include = np.array([bp in reduced_pairs for bp in bipolar_pairs])
        pow_mat =  self.get_passed_object('pow_mat')
        pow_mat = pow_mat.reshape((len(pow_mat),len(bipolar_pairs),-1))[:,to_include,:]
        return pow_mat.reshape((len(pow_mat),-1))

    def pass_objects(self):
        subject=self.pipeline.subject
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

    def get_events(self):
        return self.get_passed_object('FR1_events')


    def run(self):

        events = self.get_events()
        self.pow_mat = normalize_sessions(self.get_pow_mat(), events)

        # n1 = np.sum(events.recalled)
        # n0 = len(events) - n1
        # w0 = (2.0/n0) / ((1.0/n0)+(1.0/n1))
        # w1 = (2.0/n1) / ((1.0/n0)+(1.0/n1))

        # self.lr_classifier = LogisticRegression(C=self.params.C, penalty=self.params.penalty_type, class_weight='auto',
        #                                         solver='liblinear')
        #
        # self.lr_classifier = LogisticRegression(C=self.params.C, penalty=self.params.penalty_type, class_weight='auto',
        #                                         solver='newton-cg')

        self.lr_classifier = LogisticRegression(C=self.params.C, penalty=self.params.penalty_type, #class_weight='auto',
                                                solver='newton-cg')


        event_sessions = events.session

        recalls = events.recalled
        recalls[events.type=='REC_WORD'] = 1
        recalls[events.type=='REC_BASE'] = 0

        samples_weights = np.ones(events.shape[0], dtype=np.float)

        # samples_weights[~(events.type=='WORD')] = self.params.retrieval_samples_weight
        samples_weights[(events.type=='WORD')] = self.params.encoding_samples_weight



        sessions = np.unique(event_sessions)
        if len(sessions) > 1:
            print 'Performing permutation test'
            self.perm_AUCs = self.permuted_loso_AUCs(event_sessions, recalls, samples_weights,events=events)

            print 'Performing leave-one-session-out xval'
            self.run_loso_xval(event_sessions, recalls, permuted=False,samples_weights=samples_weights, events=events)
        else:
            sess = sessions[0]
            event_lists = events.list

            print 'Performing in-session permutation test'
            self.perm_AUCs = self.permuted_lolo_AUCs(sess, event_lists, recalls,samples_weights=samples_weights)

            print 'Performing leave-one-list-out xval'
            self.run_lolo_xval(sess, event_lists, recalls, permuted=False,samples_weights=samples_weights)

        print 'CROSS VALIDATION AUC =', self.xval_output[-1].auc

        self.pvalue = np.nansum(self.perm_AUCs >= self.xval_output[-1].auc) / float(self.perm_AUCs[~np.isnan(self.perm_AUCs)].size)
        print 'Perm test p-value =', self.pvalue


        print 'thresh =', self.xval_output[-1].jstat_thresh, 'quantile =', self.xval_output[-1].jstat_quantile



        # Finally, fitting classifier on all available data
        self.lr_classifier.fit(self.pow_mat, recalls, samples_weights)

        # FYI - in-sample AUC
        recall_prob_array = self.lr_classifier.predict_proba(self.pow_mat)[:,1]
        insample_auc = roc_auc_score(recalls, recall_prob_array)
        print 'in-sample AUC=', insample_auc

        self.pass_objects()



    def restore(self):
        subject = self.pipeline.subject

        classifier_path = self.get_path_to_resource_in_workspace(subject + '-lr_classifier.pkl')
        self.lr_classifier = joblib.load(classifier_path)
        # self.lr_classifier = joblib.load(self.get_path_to_resource_in_workspace(subject + '-lr_classifier.pkl'))
        self.xval_output = joblib.load(self.get_path_to_resource_in_workspace(subject + '-xval_output.pkl'))
        self.perm_AUCs = joblib.load(self.get_path_to_resource_in_workspace(subject + '-perm_AUCs.pkl'))
        self.pvalue = joblib.load(self.get_path_to_resource_in_workspace(subject + '-pvalue.pkl'))

        events = self.get_passed_object('FR1_events')
        probs = self.xval_output[-1].probs
        if len(probs) != len(events):
            self.run()
        else:
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
        self.lr_classifier = joblib.load(self.get_path_to_resource_in_workspace(subject+'lr_classifier_full.pkl'))
        self.pvalue = joblib.load(self.get_path_to_resource_in_workspace(subject+'-pvalue_full.pkl'))

        # self.compare_AUCs()
        self.pass_object('full_classifier_path',full_classifier_path)
        self.pass_object('xval_output_all_electrodes',self.xval_output)
        self.pass_object('lr_classifier_full',self.lr_classifier)
        self.pass_object('pvalue_full',self.pvalue)


    def pass_objects(self):
        subject=self.pipeline.subject
        classifier_path = self.get_path_to_resource_in_workspace(subject+'lr_classifier_full.pkl')
        joblib.dump(self.lr_classifier,classifier_path)
        joblib.dump(self.xval_output,self.get_path_to_resource_in_workspace(subject+'-xval_output_all_electrodes.pkl'))
        self.pass_object('full_classifier_path',classifier_path)
        self.pass_object('xval_output_all_electrodes',self.xval_output)
        joblib.dump(self.pvalue,self.get_path_to_resource_in_workspace(subject+'-pvalue_full.pkl'))
        self.pass_object('pvalue_full',self.pvalue)
        self.pass_object('lr_classifier_full',self.lr_classifier)


    def compare_AUCs(self):
        reduced_xval_output = self.get_passed_object('xval_output')
        print '\n\n'
        print 'AUC WITH ALL ELECTRODES: ', self.xval_output[-1].auc
        print 'AUC EXCLUDING STIM-ADJACENT ELECTRODES: ', reduced_xval_output[-1].auc

    def run(self):
        super(ComputeFullClassifier,self).run()
        # self.compare_AUCs()


