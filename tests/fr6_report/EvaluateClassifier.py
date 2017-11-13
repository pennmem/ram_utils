import hashlib
import classiflib
import numpy as np

from os import path
from random import shuffle
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score

from ReportUtils import  ReportRamTask
from ptsa.data.readers.IndexReader import JsonIndexReader
from ptsa.data.readers import EEGReader
from ReportTasks.RamTaskMethods import ModelOutput
from ramutils.classifier.utils import reload_classifier, get_sample_weights
from ramutils.classifier.cross_validation import permuted_lolo_AUCs


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

        experiments = ['FR1', 'catFR1'] + [task]

        for experiment in experiments:
            event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=experiment)))
            for fname in event_files:
                with open(fname,'rb') as f:
                    hash_md5.update(f.read())
        return hash_md5.digest()

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        events = self.get_passed_object(task+'_events')

        stim_off_events = self.get_passed_object('stim_off_events')
        events = np.concatenate([events,stim_off_events]).view(np.recarray)

        # Combine power matrices from stim and post_stim periods
        self.pow_mat = self.get_passed_object('fr_stim_pow_mat')
        post_stim_pow_mat = self.get_passed_object('post_stim_pow_mat')
        self.long_pow_mat = np.concatenate([self.pow_mat, post_stim_pow_mat])

        # Classifier should be evaluated on the non-stim events because the
        # expectation is that applying stimulation will affect the biomarker
        # which in turn affects the classifier
        non_stim = (events.type=='WORD') & (events.phase != 'STIM')
        all_nonstim_probs = np.array([])
        all_word_probs = np.array([])
        all_probs = np.array([])

        sessions = np.unique(events.session)
        for session in sessions:
            # We need predicted probabilities for the non-stim events separately from
            # the stim off events
            nonstim_session_events_mask = (events.session == session) & (non_stim)
            nonstim_session_events = events[nonstim_session_events_mask]
            nonstim_pow_mat = self.long_pow_mat[nonstim_session_events_mask]
            nonstim_recalls = nonstim_session_events.recalled

            # Get recorded pairs from the EEG signal. Since these EEGs are
            # used in ComputePowers, it may make sense to save out the
            # pairs by session in that task to avoid needing to re-load
            # the eeg, which is a bit slow. The alternative is to load
            # sense channels from the config file, which should be a 
            # faster process
            eeg_reader = EEGReader(events=nonstim_session_events,
                                   start_time=self.params.fr1_start_time,
                                   end_time=self.params.fr1_end_time)
            eeg = eeg_reader.read()
            recorded_channels = eeg['bipolar_pairs'].values
            recorded_channels = np.array([(int(a), int(b)) for a,b in recorded_channels])

            # Load and use the session-specific classifier
            classifier_container = reload_classifier(subject, task, session, 
                                                     mount_point=self.pipeline.mount_point)
            used_pairs = classifier_container.pairs[["contact0", "contact1"]]
            used_pairs = np.array([(int(a), int(b)) for a,b in used_pairs])

            # isin will return an array of arrays with element-wise mask, i.e. [[True, False], [True, True]]
            # We need this to be one-dimensional and should only be true if both
            # elements are true
            pair_mask = np.isin(recorded_channels, used_pairs)
            pair_mask = np.apply_along_axis(max, 1, pair_mask) # apply max function row-wise
            nonstim_pow_mat = nonstim_pow_mat.reshape((len(nonstim_pow_mat), -1, len(self.params.freqs)))
            nonstim_pow_mat = nonstim_pow_mat[:, pair_mask, :] # n_events, n_electrodes, n_frequencies
            nonstim_pow_mat = nonstim_pow_mat.reshape((len(nonstim_pow_mat),-1)) # back to 2-D

            session_specific_classifier = classifier_container.classifier
            session_specific_classifier.fit(nonstim_pow_mat, nonstim_recalls)
            nonstim_probs = session_specific_classifier.predict_proba(nonstim_pow_mat)[:, 1]
            
            all_nonstim_probs = np.append(all_nonstim_probs, nonstim_probs)

            self.xval_output[session] = ModelOutput(nonstim_recalls, nonstim_probs)
            self.xval_output[session].compute_roc()
            self.xval_output[session].compute_tercile_stats()
            self.xval_output[session].compute_normal_approx()
            
            # Always perform in-session permutation test for single-session classifier info
            self.perm_AUCs[session] = permuted_lolo_AUCs(session_specific_classifier,
                                                         nonstim_pow_mat,
                                                         nonstim_session_events,
                                                         self.params.n_perm)

            self.pvalue[session] = (np.sum(self.perm_AUCs[session] >= 
                self.xval_output[session].auc) / float(self.perm_AUCs[session].size))
            

            # Now, calculate the full set of probabilities including stim off events.
            # This is annoying because it still needs to be done on a per-session basis
            # since the classifier could be different
            session_mask = (events.session == session)
            session_specific_classifier = classifier_container.classifier
            session_specific_classifier.fit(self.long_pow_mat[session_mask], 
                                            events[session_mask].recalled)
            probs = session_specific_classifier.predict_proba(self.long_pow_mat[session_mask])[:, 1]
            all_probs = np.append(all_probs, probs)

            # Now, calculate the set of predicted probabilities for all word event to
            # be used in the ComputeFRStimTable task
            word_session_events_mask = (events.session == session) & (events.type == "WORD")
            session_specific_classifier = classifier_container.classifier
            session_specific_classifier.fit(self.long_pow_mat[word_session_events_mask], 
                                            events[word_session_events_mask].recalled)
            word_probs = session_specific_classifier.predict_proba(self.long_pow_mat[word_session_events_mask])[:, 1]
            all_word_probs = np.append(all_word_probs, word_probs)
        
        all_nonstim_probs = all_nonstim_probs.flatten()
        all_word_probs = all_word_probs.flatten()
        all_probs = all_probs.flatten()

        if not non_stim.any():
            self.xval_output = self.perm_AUCs = self.pvalue = None
        else:
            recalls = events[non_stim].recalled
            self.xval_output[-1] = ModelOutput(recalls, all_nonstim_probs)
            self.xval_output[-1].compute_roc()
            self.xval_output[-1].compute_tercile_stats()
            self.xval_output[-1].compute_normal_approx()
            print('AUC = %f' % self.xval_output[-1].auc)

            # Average over the individual sessions to get cross-session p-value and permutation AUC
            self.pvalue[-1] =  np.mean([self.pvalue[session] for session in sessions])
            # TODO: To get the permutation AUCs, we need a new procedure to appropriate apply LOSO
            # when the number of features can be different across sessions

        # Slight misnomer here; these are only pre-stim in *potential*,
        # rather than in fact. This will be corrected in ComputeFRStimTable
        pre_stim_probs = all_probs[(events.type=='WORD')]
        post_stim_probs = all_probs[(events.type == 'STIM_OFF')]

        self.pass_object(task+'_xval_output', self.xval_output)
        self.pass_object(task+'_perm_AUCs', self.perm_AUCs)
        self.pass_object(task+'_pvalue', self.pvalue)
        self.pass_object('pre_stim_probs',pre_stim_probs)
        self.pass_object('post_stim_probs',post_stim_probs)
        self.pass_object('word_probs', all_word_probs)

        joblib.dump(pre_stim_probs,self.get_path_to_resource_in_workspace('-'.join((subject,task,'pre_stim_probs.pkl'))))
        joblib.dump(post_stim_probs,self.get_path_to_resource_in_workspace('-'.join((subject,task,'post_stim_probs.pkl'))))
        joblib.dump(all_word_probs, self.get_path_to_resource_in_workspace('_'.join((subject,task,'word_probs.pkl'))))
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
        word_probs = joblib.load(self.get_path_to_resource_in_workspace('_'.join((subject,task,'word_probs.pkl'))))

        self.pass_object('pre_stim_probs',pre_stim_probs)
        self.pass_object('post_stim_probs',post_stim_probs)
        self.pass_object('word_probs', word_probs)
        self.pass_object(task + '_xval_output', self.xval_output)
        self.pass_object(task + '_perm_AUCs', self.perm_AUCs)
        self.pass_object(task + '_pvalue', self.pvalue)

        return
