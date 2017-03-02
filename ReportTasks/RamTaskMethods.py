""" Leon Davis 2/1/17
    This module collates a bunch of methods common to the RAM reporting pipeline, so that all the reports
    have a bank of shared code for their common tasks.
    I tried earlier to actually build this into the inheritance tree, but it made lining everything up sufficiently
    painful that I would have ended up producing something like this, but inheritance-based.
    Instead, I'm just going to define a number of methods and make every class call them, because I don't actually believe
    in OOP"""


from ptsa.data.readers import EEGReader
from ptsa.data.filters import MonopolarToBipolarMapper,MorletWaveletFilterCpp,MorletWaveletFilter,ButterworthFilter
import numpy as np
from scipy.stats.mstats import zscore
import time
import warnings
from math import sqrt
from random import shuffle
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.externals import joblib

def compute_powers(events,monopolar_channels,bipolar_pairs,
                   start_time,end_time,buffer_time,
                   freqs,log_powers,ComputePowers=None,filt_order=4,width=5):

    if not isinstance(bipolar_pairs,np.recarray):
        bipolar_pairs = np.array(bipolar_pairs,dtype=[('ch0','S3'),('ch1','S3')]).view(np.recarray)
    sessions = np.unique(events.session)
    pow_mat = None
    tic = time.time()
    filter_time=0.
    for sess in sessions:
        print 'Loading EEG for session {}'.format(sess)
        sess_events = events[events.session==sess]
        # Load EEG
        eeg_reader = EEGReader(events=sess_events,channels=monopolar_channels,start_time=start_time,end_time=end_time)
        eeg = eeg_reader.read()
        if eeg_reader.removed_bad_data():
            print 'REMOVED SOME BAD EVENTS !!!'
            events = np.concatenate((events[events.session !=sess],eeg['events'].data.view(np.recarray))).view(np.recarray)
            event_fields = events.dtype.names
            order = tuple(f for f in ['session','list','mstime'] if f in event_fields)
            ev_order = np.argsort(events, order=order)
            events = events[ev_order]
            #The task will have to actually handle passing the new events
        eeg=eeg.add_mirror_buffer(duration=buffer_time)

        # Use bipolar pairs
        eeg= MonopolarToBipolarMapper(time_series=eeg,bipolar_pairs=bipolar_pairs).filter()
        #Butterworth filter to remove line noise
        eeg=eeg.filtered(freq_range=[58.,62.],filt_type='stop',order=filt_order)
        print 'Computing powers'
        filter_tic=time.time()
        sess_pow_mat,phase_mat=MorletWaveletFilterCpp(time_series=eeg,freqs = freqs,output='power', width=width,
                                                      cpus=25).filter()
        print 'Total time for wavelet decomposition: %.5f s'%(time.time()-filter_tic)
        sess_pow_mat=sess_pow_mat.remove_buffer(buffer_time).data

        if log_powers:
            np.log10(sess_pow_mat,sess_pow_mat)
        sess_pow_mat = np.nanmean(sess_pow_mat.transpose(2,1,0,3),-1)

        pow_mat = sess_pow_mat if pow_mat is None else np.concatenate((pow_mat,sess_pow_mat))

    pow_mat = pow_mat.reshape((len(events),len(bipolar_pairs)*len(freqs)))
    print 'Total time elapsed: {}'.format(time.time()-tic)
    # print 'Time spent on wavelet filter: {}'.format(filter_time)
    if ComputePowers is not None:
        ComputePowers.samplerate = eeg['samplerate']
    return pow_mat,events

"""======================================== Classifier Functions =================================================== """

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


def permuted_loso_AUCs(self, event_sessions, recalls):
    n_perm = self.params.n_perm
    permuted_recalls = np.array(recalls)
    AUCs = np.empty(shape=n_perm, dtype=np.float)
    with joblib.Parallel(n_jobs=-1,verbose=20,) as parallel:
        probs = parallel(joblib.delayed(run_loso_xval)(event_sessions, permuted_recalls,
                                   self.pow_mat, self.lr_classifier,self.xval_output,
                                    permuted=True,iter=i) for i in xrange(n_perm))
        AUCs[:] = [roc_auc_score(recalls, prob) for prob in probs]
    return AUCs


def permuted_lolo_AUCs(self, events):
    n_perm = self.params.n_perm
    try:
        recalls = events.recalled
    except AttributeError:
        recalls = events.correct
    permuted_recalls = np.array(recalls)
    AUCs = np.empty(shape=n_perm, dtype=np.float)
    sessions = np.unique(events.session)
    with joblib.Parallel(n_jobs=-1,verbose=20,max_nbytes=1e4) as parallel:
        probs = parallel(joblib.delayed(run_lolo_xval)(events, permuted_recalls,self.pow_mat,self.lr_classifier,
                                                       self.xval_output, permuted=True,iter=i)
                         for i in xrange(n_perm) )
        AUCs[:] = [roc_auc_score(recalls, p) for p in probs]
    return AUCs


def run_lolo_xval(events, recalls, pow_mat,lr_classifier,xval_output, permuted=False,**kwargs):
    probs = np.empty_like(recalls, dtype=np.float)

    sessions = np.unique(events.session)

    if permuted:
        for sess in sessions:
            sess_lists = np.unique(events[events.session==sess].list)
            for lst in sess_lists:
                sel = (events.session==sess) & (events.list==lst)
                list_permuted_recalls = recalls[sel]
                shuffle(list_permuted_recalls)
                recalls[sel] = list_permuted_recalls

    for sess in sessions:
        sess_lists = np.unique(events[events.session==sess].list)
        for lst in sess_lists:
            insample_mask = (events.session!=sess) | (events.list!=lst)
            insample_pow_mat = pow_mat[insample_mask]
            insample_recalls = recalls[insample_mask]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lr_classifier.fit(insample_pow_mat, insample_recalls)

            outsample_mask = ~insample_mask
            outsample_pow_mat = pow_mat[outsample_mask]

            probs[outsample_mask] = lr_classifier.predict_proba(outsample_pow_mat)[:,1]

    if not permuted:
        xval_output[-1] = ModelOutput(recalls, probs)
        xval_output[-1].compute_roc()
        xval_output[-1].compute_tercile_stats()
        xval_output[-1].compute_normal_approx()

    return probs



def run_loso_xval(event_sessions, recalls,pow_mat,classifier,xval_output,permuted=False,**kwargs):
    if permuted:
        for sess in event_sessions:
            sel = (event_sessions == sess)
            sess_permuted_recalls = recalls[sel]
            shuffle(sess_permuted_recalls)
            recalls[sel] = sess_permuted_recalls

    probs = np.empty_like(recalls, dtype=np.float)

    sessions = np.unique(event_sessions)

    for sess in sessions:
        insample_mask = (event_sessions != sess)
        insample_pow_mat = pow_mat[insample_mask]
        insample_recalls = recalls[insample_mask]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier.fit(insample_pow_mat, insample_recalls)

        outsample_mask = ~insample_mask
        outsample_pow_mat = pow_mat[outsample_mask]
        outsample_recalls = recalls[outsample_mask]

        outsample_probs = classifier.predict_proba(outsample_pow_mat)[:,1]
        if not permuted:
            xval_output[sess] = ModelOutput(outsample_recalls, outsample_probs)
            xval_output[sess].compute_roc()
            xval_output[sess].compute_tercile_stats()
        probs[outsample_mask] = outsample_probs

    if not permuted:
        xval_output[-1] = ModelOutput(recalls, probs)
        xval_output[-1].compute_roc()
        xval_output[-1].compute_tercile_stats()
        xval_output[-1].compute_normal_approx()

    return probs