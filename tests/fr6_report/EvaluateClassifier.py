import hashlib
import classiflib
import numpy as np

from os import path
from random import shuffle
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score

from ReportUtils import  ReportRamTask
from ptsa.data.readers.IndexReader import JsonIndexReader, EEGReader
from ReportTasks.RamTaskMethods import ModelOutput
from ramutils.classifier.utils import reload_classifier


class EvaluateClassifier(ReportRamTask):
    def __init__(self, params, mark_as_completed=False):
        super(EvaluateClassifier, self).__init__(mark_as_completed=mark_as_completed)
        self.params = params
        self.pow_mat = None
        self.lr_classifier = None
        self.xval_output = dict()  # ModelOutput per session; xval_output[-1] is across all sessions
        self.perm_AUCs = {}
        self.pvalue = {}


    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()
        hash_md5.update(__file__)

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
        permuted_recalls = np.random.randint(2,size=recalls.shape)
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
        events = self.get_passed_object(task+'_events')
        monopolar_channels = self.get_passed_object("monopolar_channels")
        bipolar_channels = self.get_passed_object("bipolar_channels")

        pairs = self.get_passed_object('bipolar_pairs') # TODO: Get these from elsewhere since they do not seem to be complete
        pairs = np.array([(int(a), int(b)) for a,b in pairs])

        stim_off_events = self.get_passed_object('stim_off_events')
        events = np.concatenate([events,stim_off_events]).view(np.recarray)
        self.pow_mat = self.get_passed_object('fr_stim_pow_mat')
        post_stim_pow_mat = self.get_passed_object('post_stim_pow_mat')
        self.long_pow_mat = np.concatenate([self.pow_mat,post_stim_pow_mat])

        sessions = np.unique(events.session)
        non_stim = (events.type=='WORD') & (events.phase != 'STIM')
        all_probs = np.array([])
        for session in sessions:
            sess_events = (events.session == session) & (non_stim)
            pow_mat = self.long_pow_mat[sess_events]
            recalls = events[sess_events].recalled

            # Get recorded pairs from the EEG signal
            eeg_reader = EEGReader(events=sess_events,
                                   channels=monopolar_channels,
                                   start_time=self.params.fr1_start_time,
                                   end_time=self.params.fr1_end_time)
            eeg = eeg_reader.read()
            recorded_channels = eeg['bipolar_pairs'].values
            recorded_channels = np.array([(int(a), int(b)) for a,b in recorded_channels]) # should be same size as power matrix?

            # Load and use the session-specific classifier
            classifier_container = reload_classifier(subject, task, session, mount_point=self.pipeline.mount_point)
            used_pairs = classifier_container.pairs[["contact0", "contact1"]]
            used_pairs = np.array([(int(a), int(b)) for a,b in used_pairs])

            pair_mask = np.isin(recorded_channels, used_pairs) # mask of which pairs were actually used during the session
            pow_mat = pow_mat.reshape((len(pow_mat),-1, len(self.params.freqs)))[:,pair_mask,:]
            pow_mat.reshape((len(pow_mat),-1))

            session_specific_classifier = classifier_container.classifier
            session_specific_classifier.fit(pow_mat, recalls, classifier_container.sample_weight)

            probs = session_specific_classifier.predict_proba(pow_mat)[:, 1]
            all_probs = np.append(probs)

            self.xval_output[session] = ModelOutput(recalls, probs)
            self.xval_output[session].compute_roc()
            self.xval_output[session].compute_tercile_stats()
            self.xval_output[session].compute_normal_approx()
            # Always perform in-session permutation test for single-session classifier info

            self.perm_AUCs[session] = self.permuted_lolo_AUCs(events[sess_events])
            self.pvalue[session] = np.sum(self.perm_AUCs[session] >= self.xval_output[session].auc) / float(self.perm_AUCs[session].size)

        all_probs = all_probs.flatten()

        if not non_stim.any():
            self.xval_output = self.perm_AUCs = self.pvalue = None
        else:
            # TODO: Add Tung's AUC that is just the average of the session-specific results
            self.pow_mat = self.long_pow_mat[non_stim]
            probs = all_probs[non_stim] # concatenate after calculating them session by session
            recalls = events[non_stim].recalled
            self.xval_output[-1] = ModelOutput(recalls, probs)
            self.xval_output[-1].compute_roc()
            self.xval_output[-1].compute_tercile_stats()
            self.xval_output[-1].compute_normal_approx()

            print 'AUC = %f'%self.xval_output[-1].auc
            sessions = np.unique(events.session)
            nonstimlist_events = events[non_stim]
            if len(sessions)>1:
                print 'Performing permutation test'
                self.perm_AUCs[-1] = self.permuted_loso_AUCs(nonstimlist_events.session, recalls)
            else:
                print 'Performing in-session permutation test'
                self.perm_AUCs[-1] = self.permuted_lolo_AUCs(nonstimlist_events)
            self.pvalue[-1] = np.sum(self.perm_AUCs[-1] >= self.xval_output[-1].auc) / float(self.perm_AUCs[-1].size)
            print 'Perm test p-value = ', self.pvalue

        # Slight misnomer here; these are only pre-stim in *potential*,
        # rather than in fact. This will be corrected in ComputeFRStimTable
        pre_stim_probs = all_probs[(events.type=='WORD')]
        post_stim_probs = all_probs[events.type=='STIM_OFF']

        self.pass_object(task+'_xval_output', self.xval_output)
        self.pass_object(task+'_perm_AUCs', self.perm_AUCs)
        self.pass_object(task+'_pvalue', self.pvalue)
        self.pass_object('pre_stim_probs',pre_stim_probs)
        self.pass_object('post_stim_probs',post_stim_probs)

        joblib.dump(pre_stim_probs,self.get_path_to_resource_in_workspace('-'.join((subject,task,'pre_stim_probs.pkl'))))
        joblib.dump(post_stim_probs,self.get_path_to_resource_in_workspace('-'.join((subject,task,'post_stim_probs.pkl'))))
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
        pre_stim_probs = joblib.load(self.get_path_to_resource_in_workspace('-'.join((subject,task,'pre_stim_probs.pkl'))))
        post_stim_probs = joblib.load(self.get_path_to_resource_in_workspace('-'.join((subject,task,'post_stim_probs.pkl'))))

        self.pass_object('pre_stim_probs',pre_stim_probs)
        self.pass_object('post_stim_probs',post_stim_probs)
        self.pass_object(task + '_xval_output', self.xval_output)
        self.pass_object(task + '_perm_AUCs', self.perm_AUCs)
        self.pass_object(task + '_pvalue', self.pvalue)
