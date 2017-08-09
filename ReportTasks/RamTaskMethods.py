""" Leon Davis 2/1/17
    This module collates a bunch of methods common to the RAM reporting pipeline, so that all the reports
    have a bank of shared code for their common tasks.
    I tried earlier to actually build this into the inheritance tree, but it made lining everything up sufficiently
    painful that I would have ended up producing something like this, but inheritance-based.
    Instead, I'm just going to define a number of methods and make every class call them, because I don't actually believe
    in OOP"""

from ptsa.data.readers import EEGReader
from ptsa.data.filters import MonopolarToBipolarMapper, MorletWaveletFilterCpp, MorletWaveletFilter, ButterworthFilter
import numpy as np
from scipy.stats.mstats import zscore
import time
import warnings
from math import sqrt
from random import shuffle
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.externals import joblib



def compute_wavelets_powers(events, monopolar_channels, bipolar_pairs,
                   start_time, end_time, buffer_time,
                   freqs, filt_order=4, width=5):
    if not isinstance(bipolar_pairs, np.recarray):
        # it expects to receive a list
        bipolar_pairs = np.array(bipolar_pairs, dtype=[('ch0', 'S3'), ('ch1', 'S3')]).view(np.recarray)
    else:
        # to get the same treatment if we get recarray , we will convert it to a list and then bask to
        # recarray with correct dtype
        bipolar_pairs = np.array(list(bipolar_pairs), dtype=[('ch0', 'S3'), ('ch1', 'S3')]).view(np.recarray)

    sessions = np.unique(events.session)
    pow_mat = None
    tic = time.time()
    filter_time = 0.

    sess_events = events
    # Load EEG
    eeg_reader = EEGReader(events=sess_events, channels=monopolar_channels, start_time=start_time,
                           end_time=end_time)
    eeg = eeg_reader.read()
    samplerate = eeg['samplerate']

    eeg = eeg.add_mirror_buffer(duration=buffer_time)

    # Use bipolar pairs
    eeg = MonopolarToBipolarMapper(time_series=eeg, bipolar_pairs=bipolar_pairs).filter()
    # Butterworth filter to remove line noise
    eeg = eeg.filtered(freq_range=[58., 62.], filt_type='stop', order=filt_order)
    eeg['samplerate'] = samplerate

    print 'Computing powers'

    filter_tic = time.time()

    # making underlying array contiguous
    eeg.data = np.ascontiguousarray(eeg.data)

    wavelet_filter = MorletWaveletFilterCpp(time_series=eeg, freqs=freqs, output='power', width=width,
                                            cpus=25)

    sess_pow_mat, phase_mat = wavelet_filter.filter()

    print 'Total time for wavelet decomposition: %.5f s' % (time.time() - filter_tic)
    sess_pow_mat = sess_pow_mat.remove_buffer(buffer_time).data

    return sess_pow_mat



# def compute_wavelets_powers(events, monopolar_channels, bipolar_pairs,
#                    start_time, end_time, buffer_time,
#                    freqs, filt_order=4, width=5):
#     if not isinstance(bipolar_pairs, np.recarray):
#         # it expects to receive a list
#         bipolar_pairs = np.array(bipolar_pairs, dtype=[('ch0', 'S3'), ('ch1', 'S3')]).view(np.recarray)
#     else:
#         # to get the same treatment if we get recarray , we will convert it to a list and then bask to
#         # recarray with correct dtype
#         bipolar_pairs = np.array(list(bipolar_pairs), dtype=[('ch0', 'S3'), ('ch1', 'S3')]).view(np.recarray)
#
#     sessions = np.unique(events.session)
#     pow_mat = None
#     tic = time.time()
#     filter_time = 0.
#     for sess in sessions:
#         print 'Loading EEG for session {}'.format(sess)
#         sess_events = events[events.session == sess]
#         # Load EEG
#         eeg_reader = EEGReader(events=sess_events, channels=monopolar_channels, start_time=start_time,
#                                end_time=end_time)
#         eeg = eeg_reader.read()
#         samplerate = eeg['samplerate']
#         if eeg_reader.removed_bad_data():
#             print 'REMOVED SOME BAD EVENTS !!!'
#             events = np.concatenate((events[events.session != sess], eeg['events'].data.view(np.recarray))).view(
#                 np.recarray)
#             event_fields = events.dtype.names
#             order = tuple(f for f in ['session', 'list', 'mstime'] if f in event_fields)
#             ev_order = np.argsort(events, order=order)
#             events = events[ev_order]
#             # The task will have to actually handle passing the new events
#         eeg = eeg.add_mirror_buffer(duration=buffer_time)
#
#         # Use bipolar pairs
#         eeg = MonopolarToBipolarMapper(time_series=eeg, bipolar_pairs=bipolar_pairs).filter()
#         # Butterworth filter to remove line noise
#         eeg = eeg.filtered(freq_range=[58., 62.], filt_type='stop', order=filt_order)
#         eeg['samplerate'] = samplerate
#         print 'Computing powers'
#         filter_tic = time.time()
#
#         # making underlying array contiguous
#         eeg.data = np.ascontiguousarray(eeg.data)
#
#         wavelet_filter = MorletWaveletFilterCpp(time_series=eeg, freqs=freqs, output='power', width=width,
#                                                 cpus=25)
#
#         sess_pow_mat, phase_mat = wavelet_filter.filter()
#
#         print 'Total time for wavelet decomposition: %.5f s' % (time.time() - filter_tic)
#         sess_pow_mat = sess_pow_mat.remove_buffer(buffer_time).data
#
#     return sess_pow_mat




def compute_powers(events, monopolar_channels, bipolar_pairs,
                   start_time, end_time, buffer_time,
                   freqs, log_powers, ComputePowers=None, filt_order=4, width=5):
    if not isinstance(bipolar_pairs, np.recarray):
        # it expects to receive a list
        bipolar_pairs = np.array(bipolar_pairs, dtype=[('ch0', 'S3'), ('ch1', 'S3')]).view(np.recarray)
    else:
        # to get the same treatment if we get recarray , we will convert it to a list and then bask to
        # recarray with correct dtype
        bipolar_pairs = np.array(list(bipolar_pairs), dtype=[('ch0', 'S3'), ('ch1', 'S3')]).view(np.recarray)

    sessions = np.unique(events.session)
    pow_mat = None
    tic = time.time()
    filter_time = 0.
    for sess in sessions:
        print 'Loading EEG for session {}'.format(sess)
        sess_events = events[events.session == sess]
        # Load EEG
        eeg_reader = EEGReader(events=sess_events, channels=monopolar_channels, start_time=start_time,
                               end_time=end_time)
        eeg = eeg_reader.read()
        samplerate = eeg['samplerate']
        if eeg_reader.removed_bad_data():
            print 'REMOVED SOME BAD EVENTS !!!'
            events = np.concatenate((events[events.session != sess], eeg['events'].data.view(np.recarray))).view(
                np.recarray)
            event_fields = events.dtype.names
            order = tuple(f for f in ['session', 'list', 'mstime'] if f in event_fields)
            ev_order = np.argsort(events, order=order)
            events = events[ev_order]
            # The task will have to actually handle passing the new events
        eeg = eeg.add_mirror_buffer(duration=buffer_time)

        # Use bipolar pairs
        eeg = MonopolarToBipolarMapper(time_series=eeg, bipolar_pairs=bipolar_pairs).filter()
        # Butterworth filter to remove line noise
        eeg = eeg.filtered(freq_range=[58., 62.], filt_type='stop', order=filt_order)
        eeg['samplerate'] = samplerate
        print 'Computing powers'
        filter_tic = time.time()

        # making underlying array contiguous
        eeg.data = np.ascontiguousarray(eeg.data)

        wavelet_filter = MorletWaveletFilterCpp(time_series=eeg, freqs=freqs, output='power', width=width,
                                                cpus=25)

        sess_pow_mat, phase_mat = wavelet_filter.filter()

        print 'Total time for wavelet decomposition: %.5f s' % (time.time() - filter_tic)
        sess_pow_mat = sess_pow_mat.remove_buffer(buffer_time).data + np.finfo(np.float).eps/2.

        if log_powers:
            np.log10(sess_pow_mat, sess_pow_mat)
        sess_pow_mat = np.nanmean(sess_pow_mat.transpose(2, 1, 0, 3), -1)

        pow_mat = sess_pow_mat if pow_mat is None else np.concatenate((pow_mat, sess_pow_mat))


    pow_mat = pow_mat.reshape((len(events), len(bipolar_pairs) * len(freqs)))

    print 'Total time elapsed: {}'.format(time.time() - tic)
    # print 'Time spent on wavelet filter: {}'.format(filter_time)
    if ComputePowers is not None:
        ComputePowers.samplerate = eeg['samplerate'].data.astype(np.float)
    return pow_mat, events


# """======================================== Classifier Functions =================================================== """


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
        # self.std1 = np.nan
        self.n0 = np.nan
        self.mean0 = np.nan
        # self.std0 = np.nan
        self.pooled_std = np.nan

    def compute_normal_approx(self):
        class1_mask = (self.true_labels == 1)
        class1_probs = self.probs[class1_mask]
        self.n1 = len(class1_probs)
        class1_normal = np.log(class1_probs / (1.0 - class1_probs))
        self.mean1 = np.mean(class1_normal)
        #self.std1 = np.std(class1_normal, ddof=1)
        var1 = np.var(class1_normal, ddof=1)
        print 'Positive class: mean =', self.mean1, 'variance =', var1, 'n =', self.n1

        class0_probs = self.probs[~class1_mask]
        self.n0 = len(class0_probs)
        class0_normal = np.log(class0_probs / (1.0 - class0_probs))
        self.mean0 = np.mean(class0_normal)
        # self.std0 = np.std(class0_normal, ddof=1)
        var0 = np.var(class0_normal, ddof=1)
        print 'Negative class: mean =', self.mean0, 'variance =', var0, 'n =', self.n0

        self.pooled_std = sqrt((var1 * (self.n1 - 1) + var0 * (self.n0 - 1)) / (self.n1 + self.n0 - 2))

        # if self.mean1 < self.mean0:
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


def permuted_loso_AUCs(self, event_sessions, recalls):
    n_perm = self.params.n_perm
    permuted_recalls = np.array(recalls)
    AUCs = np.empty(shape=n_perm, dtype=np.float)
    try:
        parallelize = self.params.parallelize
    except AttributeError:
        parallelize = True
    if parallelize:
        with joblib.Parallel(n_jobs=-1, verbose=20, ) as parallel:
            probs = parallel(joblib.delayed(run_loso_xval)(event_sessions, permuted_recalls,
                                                           self.pow_mat, self.lr_classifier, self.xval_output,
                                                           permuted=True, iter=i) for i in xrange(n_perm))
            AUCs[:] = [roc_auc_score(recalls, p) for p in probs]
    else:
        for i in range(n_perm):
            probs  = run_loso_xval(event_sessions,permuted_recalls,
                                   self.pow_mat,self.lr_classifier,self.xval_output,permuted=True)
            auc = roc_auc_score(recalls,probs)
            print("AUC = %s"%str(auc))
            AUCs[i] = auc

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
    try:
        parallelize = self.params.parallelize
    except AttributeError:
        parallelize = True
    if parallelize:
        with joblib.Parallel(n_jobs=-1, verbose=20, max_nbytes=1e4) as parallel:
            probs = parallel(joblib.delayed(run_lolo_xval)(events, permuted_recalls, self.pow_mat, self.lr_classifier,
                                                           self.xval_output, permuted=True, iter=i)
                             for i in xrange(n_perm))
            AUCs[:] = [roc_auc_score(recalls, p) for p in probs]
    else:
        for i in range(n_perm):
            probs  = run_lolo_xval(events,permuted_recalls,
                                   self.pow_mat,self.lr_classifier,self.xval_output,permuted=True)
            auc = roc_auc_score(recalls,probs)
            print("AUC = %s"%str(auc))
            AUCs[i] = auc
    return AUCs


def run_lolo_xval(events, recalls, pow_mat, lr_classifier, xval_output, permuted=False, **kwargs):
    probs = np.empty_like(recalls, dtype=np.float)

    sessions = np.unique(events.session)

    if 'list' in events.dtype.names:
        trial_type='list'
    elif 'trial' in events.dtype.names:
        trial_type='trial'
    else:
        raise RuntimeError('Unknown trial type')

    if permuted:
        for sess in sessions:
            sess_lists = np.unique(events[events.session==sess][trial_type])
            for lst in sess_lists:
                sel = (events.session==sess) & (events[trial_type]==lst)
                list_permuted_recalls = recalls[sel]
                shuffle(list_permuted_recalls)
                recalls[sel] = list_permuted_recalls

    for sess in sessions:
        sess_lists = np.unique(events[events.session==sess][trial_type])
        for lst in sess_lists:
            insample_mask = (events.session!=sess) | (events[trial_type]!=lst)
            insample_pow_mat = pow_mat[insample_mask]
            insample_recalls = recalls[insample_mask]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lr_classifier.fit(insample_pow_mat, insample_recalls)

            outsample_mask = ~insample_mask
            outsample_pow_mat = pow_mat[outsample_mask]

            probs[outsample_mask] = lr_classifier.predict_proba(outsample_pow_mat)[:, 1]
            if not permuted:
                xval_output[sess] = ModelOutput(recalls[outsample_mask], probs[outsample_mask])
                xval_output[sess].compute_roc()
                xval_output[sess].compute_tercile_stats()

    if not permuted:
        xval_output[-1] = ModelOutput(recalls, probs)
        xval_output[-1].compute_roc()
        xval_output[-1].compute_tercile_stats()
        xval_output[-1].compute_normal_approx()

    return probs


def run_loso_xval(event_sessions, recalls, pow_mat, classifier, xval_output, permuted=False, **kwargs):
    sessions = np.unique(event_sessions)
    probs = np.empty_like(recalls, dtype=np.float)
    for sess in sessions:
        if permuted:
            sel = (event_sessions == sess)
            sess_permuted_recalls = recalls[sel]
            shuffle(sess_permuted_recalls)
            recalls[sel] = sess_permuted_recalls

        insample_mask = (event_sessions != sess)
        insample_pow_mat = pow_mat[insample_mask]
        insample_recalls = recalls[insample_mask]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier.fit(insample_pow_mat, insample_recalls)

        outsample_mask = ~insample_mask
        outsample_pow_mat = pow_mat[outsample_mask]
        outsample_recalls = recalls[outsample_mask]

        outsample_probs = classifier.predict_proba(outsample_pow_mat)[:, 1]
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


"============================================================"


def free_epochs(times, duration, pre, post, start=None, end=None):
    # (list(vector(int))*int*int*int) -> list(vector(int))
    """
    Given a list of event times, find epochs between them when nothing is happening

    Parameters:
    -----------

    times:
        An iterable of 1-d numpy arrays, each of which indicates event times

    duration: int
        The length of the desired empty epochs

    pre: int
        the time before each event to exclude

    post: int
        The time after each event to exclude

    """
    n_trials = len(times)
    epoch_times = []
    for i in range(n_trials):
        ext_times = times[i]
        if start is not None:
            ext_times = np.append([start[i]], ext_times)
        if end is not None:
            ext_times = np.append(ext_times, [end[i]])
        pre_times = ext_times - pre
        post_times = ext_times + post
        interval_durations = pre_times[1:] - post_times[:-1]
        free_intervals = np.where(interval_durations > duration)[0]
        trial_epoch_times = []
        for interval in free_intervals:
            begin = post_times[interval]
            finish = pre_times[interval + 1] - duration
            interval_epoch_times = range(int(begin), int(finish), int(duration))
            trial_epoch_times.extend(interval_epoch_times)
        epoch_times.append(np.array(trial_epoch_times))
    epoch_array = np.empty((n_trials, max([len(x) for x in epoch_times])))
    epoch_array[...] = -np.inf
    for i, epoch in enumerate(epoch_times):
        epoch_array[i, :len(epoch)] = epoch
    return epoch_array


def create_baseline_events(events, start_time, end_time):
    '''
    Match recall events to matching baseline periods of failure to recall.
    Baseline events all begin at least 1000 ms after a vocalization, and end at least 1000 ms before a vocalization.
    Each recall event is matched, wherever possible, to a valid baseline period from a different list within 3 seconds
     relative to the onset of the recall period.

    Parameters:
    -----------
    events: The event structure in which to incorporate these baseline periods
    start_time: The amount of time to skip at the beginning of the session (ms)
    end_time: The amount of time within the recall period to consider (ms)

    '''

    all_events = []
    for session in np.unique(events.session):
        sess_events = events[(events.session == session)]
        irts = np.append([0], np.diff(sess_events.mstime))
        rec_events = sess_events[(sess_events.type == 'REC_WORD') & (sess_events.intrusion == 0) & (irts > 1000)]
        voc_events = sess_events[((sess_events.type == 'REC_WORD') | (sess_events.type == 'REC_WORD_VV'))]
        starts = sess_events[(sess_events.type == 'REC_START')]
        ends = sess_events[(sess_events.type == 'REC_END')]
        rec_lists = tuple(np.unique(starts.list))
        times = [voc_events[(voc_events.list == lst)].mstime if (voc_events.list==lst).any() else []
                 for lst in rec_lists]
        start_times = starts.mstime.astype(np.int)
        end_times = ends.mstime.astype(np.int)
        epochs = free_epochs(times, 500, 2000, 1000, start=start_times, end=end_times)
        rel_times = [(t - i)[(t - i > start_time) & (t - i < end_time)] for (t, i) in
                     zip([rec_events[rec_events.list == lst].mstime for lst in rec_lists ], start_times)
                     ]
        rel_epochs = epochs - start_times[:, None]
        full_match_accum = np.zeros(epochs.shape, dtype=np.bool)
        for (i, rec_times_list) in enumerate(rel_times):
            is_match = np.empty(epochs.shape, dtype=np.bool)
            is_match[...] = False
            for t in rec_times_list:
                is_match_tmp = np.abs((rel_epochs - t)) < 3000
                is_match_tmp[i, ...] = False
                good_locs = np.where(is_match_tmp & (~full_match_accum))
                if len(good_locs[0]):
                    choice_position = np.argmin(np.mod(good_locs[0] - i, len(good_locs[0])))
                    choice_inds = (good_locs[0][choice_position], good_locs[1][choice_position])
                    full_match_accum[choice_inds] = True
        matching_epochs = epochs[full_match_accum]
        new_events = np.zeros(len(matching_epochs), dtype=sess_events.dtype).view(np.recarray)
        for i, _ in enumerate(new_events):
            new_events[i].mstime = matching_epochs[i]
            new_events[i].type = 'REC_BASE'
        new_events.recalled = 0
        merged_events = np.concatenate((sess_events, new_events)).view(np.recarray)
        merged_events.sort(order='mstime')
        for (i, event) in enumerate(merged_events):
            if event.type == 'REC_BASE':
                merged_events[i].session = merged_events[i - 1].session
                merged_events[i].list = merged_events[i - 1].list
                merged_events[i].eegfile = merged_events[i - 1].eegfile
                merged_events[i].eegoffset = merged_events[i - 1].eegoffset + (
                merged_events[i].mstime - merged_events[i - 1].mstime)
        all_events.append(merged_events)
    return np.concatenate(all_events).view(np.recarray)


def create_baseline_events_pal(events):
    '''
    Match recall events to matching baseline periods of failure to recall.
    Baseline events all begin at least 1000 ms after a vocalization, and end at least 1000 ms before a vocalization.
    Each recall event is matched, wherever possible, to a valid baseline period from a different list within 3 seconds
     relative to the onset of the recall period.

    Parameters:
    -----------
    events: The event structure in which to incorporate these baseline periods

    '''

    all_events = []
    for session in np.unique(events.session):
        sess_events = events[(events.session == session)]
        irts = np.append([0], np.diff(sess_events.mstime))
        rec_events = sess_events[(sess_events.type == 'REC_EVENT') & (sess_events.intrusion == 0) & (irts > 1000)]
        voc_events = sess_events[((sess_events.type == 'REC_EVENT') | (sess_events.type == 'REC_EVENT_VV'))]
        starts = sess_events[(sess_events.type == 'RECALL_START')]
        ends = sess_events[(sess_events.type == 'RECALL_END')]
        rec_lists = tuple(np.unique(starts.list))
        times = [voc_events[(voc_events.list == lst)].mstime for lst in rec_lists]
        start_times = starts.mstime
        end_times = ends.mstime
        epochs = free_epochs(times, 500, 1000, 1000, start=start_times, end=end_times)
        rel_times = [t - i for (t, i) in
                     zip([rec_events[rec_events.list == lst].mstime for lst in rec_lists], start_times)]
        rel_epochs = epochs - start_times[:, None]
        full_match_accum = np.zeros(epochs.shape, dtype=np.bool)
        for (i, rec_times_list) in enumerate(rel_times):
            is_match = np.empty(epochs.shape, dtype=np.bool)
            is_match[...] = False
            for t in rec_times_list:
                is_match_tmp = np.abs((rel_epochs - t)) < 3000
                is_match_tmp[i, ...] = False
                good_locs = np.where(is_match_tmp & (~full_match_accum))
                if len(good_locs[0]):
                    choice_position = np.argmin(np.mod(good_locs[0] - i, len(good_locs[0])))
                    choice_inds = (good_locs[0][choice_position], good_locs[1][choice_position])
                    full_match_accum[choice_inds] = True
        matching_epochs = epochs[full_match_accum]
        new_events = np.zeros(len(matching_epochs), dtype=sess_events.dtype).view(np.recarray)
        for i, _ in enumerate(new_events):
            new_events[i].mstime = matching_epochs[i]
            new_events[i].type = 'REC_BASE'
        new_events.recalled = 0
        merged_events = np.concatenate((sess_events, new_events)).view(np.recarray)
        merged_events.sort(order='mstime')
        for (i, event) in enumerate(merged_events):
            if event.type == 'REC_BASE':
                merged_events[i].session = merged_events[i - 1].session
                merged_events[i].list = merged_events[i - 1].list
                merged_events[i].eegfile = merged_events[i - 1].eegfile
                merged_events[i].eegoffset = merged_events[i - 1].eegoffset + (
                merged_events[i].mstime - merged_events[i - 1].mstime)
        all_events.append(merged_events)
    return np.concatenate(all_events).view(np.recarray)
