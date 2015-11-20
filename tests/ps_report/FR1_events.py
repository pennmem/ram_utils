from sys import argv
import os
import os.path
import re
import numpy as np
from scipy.io import loadmat
from scipy.stats.mstats import zscore

from ptsa.data.events import Events
from ptsa.data.rawbinwrapper import RawBinWrapper

from get_bipolar_subj_elecs import get_bipolar_subj_elecs

from ptsa.wavelet import phase_pow_multi

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib


dtypes = [('subject','|S12'), ('session',np.int), ('experiment','|S12'), ('list',np.int),
          ('serialpos', np.int), ('type', '|S20'), ('item','|S20'),
          ('itemno',np.int), ('recalled',np.int),
          ('amplitude',np.float), ('burst_i',np.int), ('pulse_frequency',np.int),
          ('burst_frequency',np.int), ('nBursts', np.int), ('pulse_duration', np.int),
          ('mstime',np.float), ('rectime',np.int), ('intrusion',np.int),
          ('isStim', np.int), ('category','|S20'), ('categoryNum', np.int),
          ('stimAnode', np.int), ('stimAnodeTag','|S10'),
          ('stimCathode', np.int), ('stimCathodeTag', '|S10'),
          ('eegfile','|S256'), ('eegoffset', np.int)]


class Params:
    subject = 'R1056M'
    path_prefix = '/Volumes/RHINO'

    fr1_start_time = -0.5
    fr1_end_time = 2.1
    fr1_buf = 1.0

    ps_pre_start_time = -1.0
    ps_pre_end_time = 0.0
    ps_pre_buf = 1.0

    ps_post_offset = 0.1

    # eeghz = 500
    # powhz = 50
    freqs = np.logspace(np.log10(3), np.log10(180), 12)


def get_events(subject, task, path_prefix):
    event_file = os.path.join(path_prefix + '/data', 'events', task, subject + '_events.mat')
    events = loadmat(event_file, struct_as_record=True, squeeze_me=True)['events']
    new_events = np.rec.recarray(len(events), dtype=dtypes)
    for field in events.dtype.names:
        try:
            new_events[field] = events[field]
        except ValueError:
            print 'ValueError: field =', field
    return new_events

'/data/eeg/R1056M/eeg.reref/R1056M_19Jun15_1003'
def attach_raw_bin_wrappers(events):
    eegfiles = np.unique(events.eegfile)
    events = events.add_fields(esrc=np.dtype(RawBinWrapper))
    for eegfile in eegfiles:
        raw_bin_wrapper = RawBinWrapper(Params.path_prefix+eegfile)
        # events[events.eegfile == eegfile]['esrc'] = raw_bin_wrapper does NOT work!
        inds = np.where(events.eegfile == eegfile)[0]
        for i in inds:
            events[i]['esrc'] = raw_bin_wrapper
    return events


def get_dataroot(events):
    dataroots = np.unique([esrc.dataroot for esrc in events.esrc])
    if len(dataroots) != 1:
        raise ValueError('Invalid number of dataroots: %d' % len(dataroots))
    return dataroots[0]


def correct_eegfile_field(events):
    events = events[events.eegfile != '[]']  # remove events with no recording
    data_dir_bad = r'/data.*/' + events[0].subject + r'/eeg'
    data_dir_good = r'/data/eeg/' + events[0].subject + r'/eeg'
    for ev in events:
        ev.eegfile = ev.eegfile.replace('eeg.reref', 'eeg.noreref')
        ev.eegfile = re.sub(data_dir_bad, data_dir_good, ev.eegfile)
    return events


def get_bps(events):
    dataroot = get_dataroot(events)
    subjpath = os.path.dirname(os.path.dirname(dataroot))
    return get_bipolar_subj_elecs(subjpath, leadsonly=True, exclude_bad_leads=False)


def get_single_elecs_from_bps(tal_info):
    channels = np.array([], dtype=np.dtype('|S32'))
    for ti in tal_info:
        channels = np.hstack((channels, ti['channel_str']))
    return np.unique(channels)


def compute_fr1_powers(events, sessions, channels, tal_info):
    n_freqs = len(Params.freqs)
    n_bps = len(tal_info)

    pow_mat = None
    recalls = None

    for sess in sessions:
        sess_events = events[events.session == sess]
        n_events = len(sess_events)

        sess_recalls = sess_events.recalled

        print 'Loading EEG for', n_events, 'events of session', sess

        eegs = sess_events.get_data(channels=channels, start_time=Params.fr1_start_time, end_time=Params.fr1_end_time,
                                    buffer_time=Params.fr1_buf, eoffset='eegoffset', keep_buffer=True, eoffset_in_time=False)

        print 'Computing FR1 powers'

        sess_pow_mat = np.empty(shape=(n_events, n_bps, n_freqs), dtype=np.float)

        for i,ti in enumerate(tal_info):
            bp = ti['channel_str']
            print bp
            elec1 = np.where(channels == bp[0])[0][0]
            elec2 = np.where(channels == bp[1])[0][0]
            bp_data = eegs[elec1] - eegs[elec2]
            bp_data = bp_data.filtered([58,62], filt_type='stop', order=1)
            for ev in xrange(n_events):
                pow_ev = phase_pow_multi(Params.freqs, bp_data[ev], to_return='power')
                pow_ev = pow_ev.remove_buffer(Params.fr1_buf)
                pow_ev = np.nanmean(pow_ev, 1)
                sess_pow_mat[ev,i,:] = pow_ev

        sess_pow_mat = sess_pow_mat.reshape((n_events, n_bps*n_freqs))
        sess_pow_mat = zscore(sess_pow_mat, axis=0, ddof=1)

        pow_mat = np.vstack((pow_mat,sess_pow_mat)) if pow_mat is not None else sess_pow_mat
        recalls = np.hstack((recalls,sess_recalls)) if recalls is not None else sess_recalls

    return pow_mat, recalls


def compute_ps_powers(events, sessions, channels, tal_info, experiment):
    n_freqs = len(Params.freqs)
    n_bps = len(tal_info)

    pow_mat_pre = pow_mat_post = None

    for sess in sessions:
        sess_events = events[events.session == sess]
        n_events = len(sess_events)

        print 'Loading EEG for', n_events, 'events of session', sess

        eegs_pre = sess_events.get_data(channels=channels, start_time=Params.ps_pre_start_time, end_time=Params.ps_pre_end_time,
                                    buffer_time=Params.ps_pre_buf, eoffset='eegoffset', keep_buffer=True, eoffset_in_time=False)

        eegs_post = np.empty_like(eegs_pre)
        post_start_time = Params.ps_post_offset
        post_end_time = Params.ps_post_offset + (Params.ps_pre_end_time - Params.ps_pre_start_time)
        for i_ev in xrange(n_events):
            ev_offset = sess_events[i_ev].pulse_duration
            if ev_offset > 0:
                if experiment == 'PS3' and sess_events[i_ev].nBursts > 0:
                    ev_offset *= sess_events[i_ev].nBursts + 1
                ev_offset *= 0.001
            else:
                ev_offset = 0.0
            eegs_post[:,i_ev:i_ev+1,:] = sess_events[i_ev:i_ev+1].get_data(channels=channels, start_time=post_start_time+ev_offset,
                        end_time=post_end_time+ev_offset, buffer_time=Params.ps_pre_buf, eoffset='eegoffset', keep_buffer=True, eoffset_in_time=False)


        print 'Computing', experiment, 'powers'

        sess_pow_mat_pre = np.empty(shape=(n_events, n_bps, n_freqs), dtype=np.float)
        sess_pow_mat_post = np.empty_like(sess_pow_mat_pre)

        for i,ti in enumerate(tal_info):
            bp = ti['channel_str']
            print bp
            elec1 = np.where(channels == bp[0])[0][0]
            elec2 = np.where(channels == bp[1])[0][0]

            bp_data_pre = eegs_pre[elec1] - eegs_pre[elec2]
            bp_data_pre = bp_data_pre.filtered([58,62], filt_type='stop', order=1)
            for ev in xrange(n_events):
                pow_pre_ev = phase_pow_multi(Params.freqs, bp_data_pre[ev], to_return='power')
                pow_pre_ev = pow_pre_ev.remove_buffer(Params.ps_pre_buf)
                pow_pre_ev = np.nanmean(pow_pre_ev, 1)
                sess_pow_mat_pre[ev,i,:] = pow_pre_ev

            bp_data_post = eegs_post[elec1] - eegs_post[elec2]
            bp_data_post = bp_data_post.filtered([58,62], filt_type='stop', order=1)
            for ev in xrange(n_events):
                pow_post_ev = phase_pow_multi(Params.freqs, bp_data_post[ev], to_return='power')
                pow_post_ev = pow_post_ev.remove_buffer(Params.ps_pre_buf)
                pow_post_ev = np.nanmean(pow_post_ev, 1)
                sess_pow_mat_post[ev,i,:] = pow_post_ev

        sess_pow_mat_pre = sess_pow_mat_pre.reshape((n_events, n_bps*n_freqs))
        sess_pow_mat_pre = zscore(sess_pow_mat_pre, axis=0, ddof=1)

        sess_pow_mat_post = sess_pow_mat_post.reshape((n_events, n_bps*n_freqs))
        sess_pow_mat_post = zscore(sess_pow_mat_post, axis=0, ddof=1)

        pow_mat_pre = np.vstack((pow_mat_pre,sess_pow_mat_pre)) if pow_mat_pre is not None else sess_pow_mat_pre
        pow_mat_post = np.vstack((pow_mat_post,sess_pow_mat_post)) if pow_mat_post is not None else sess_pow_mat_post

    return pow_mat_pre, pow_mat_post


def compute_classifier(pow_mat, recalls):
    print 'Computing logistic regression:', pow_mat.shape[0], 'samples', pow_mat.shape[1], 'features'

    lr_classifier = LogisticRegressionCV(penalty='l1', solver='liblinear')
    lr_classifier.fit(pow_mat, recalls)
    probs = lr_classifier.predict_proba(pow_mat)[:,1]
    auc = roc_auc_score(recalls, probs)

    print 'AUC =', auc

    return lr_classifier


def is_stim_event_type(event_type):
    return event_type in ['STIMULATING', 'BEGIN_BURST', 'STIM_SINGLE_PULSE']


def compute_isi(events):
    print 'Computing ISI'

    events = events.add_fields(isi=np.float)
    events.isi = np.nan

    for i in xrange(1,len(events)):
        curr_ev = events[i]
        if is_stim_event_type(curr_ev.type):
            prev_ev = events[i-1]
            if curr_ev.session == prev_ev.session:
                if is_stim_event_type(prev_ev.type) or prev_ev.type == 'BURST':
                    prev_mstime = prev_ev.mstime
                    if prev_ev.pulse_duration > 0:
                        prev_mstime += prev_ev.pulse_duration
                    curr_ev.isi = curr_ev.mstime - prev_mstime


def compute_prob_deltas(ps_pow_mat_pre, ps_pow_mat_post, lr_classifier):
    prob_pre = lr_classifier.predict_proba(ps_pow_mat_pre)[:,1]
    prob_post = lr_classifier.predict_proba(ps_pow_mat_post)[:,1]
    return prob_pre, prob_post - prob_pre


def main():
    subject = argv[1] if len(argv) > 1 else Params.subject
    experiment = argv[2] if len(argv) > 2 else 'PS2'

    print 'Requested', experiment, 'report for subject', subject

    fr1_events = Events(get_events(subject=subject, task='RAM_FR1', path_prefix=Params.path_prefix))

    fr1_events = correct_eegfile_field(fr1_events)
    fr1_events = attach_raw_bin_wrappers(fr1_events)

    tal_info = get_bps(fr1_events)
    channels = get_single_elecs_from_bps(tal_info)
    print len(channels), 'single electrodes', len(tal_info), 'bipolar pairs'

    fr1_sessions = np.unique(fr1_events.session)
    print 'FR1 sessions:', fr1_sessions

    fr1_events = fr1_events[fr1_events.type == 'WORD']
    print len(fr1_events), 'FR1 WORD events'

    fr1_pow_mat, recalls = compute_fr1_powers(fr1_events, fr1_sessions, channels, tal_info)

    joblib.dump(fr1_pow_mat, subject+'_fr1_pow_mat.pkl')
    joblib.dump(recalls, subject+'_recalls.pkl')

    # fr1_pow_mat = joblib.load(subject+'_pow_mat.pkl')
    # recalls = joblib.load(subject+'_recalls.pkl')

    lr_classifier = compute_classifier(fr1_pow_mat, recalls)

    joblib.dump(lr_classifier, subject+'_lr.pkl')

    # lr_classifier = joblib.load(subject+'_lr.pkl')

    ps_events = Events(get_events(subject=subject, task='RAM_PS', path_prefix=Params.path_prefix))
    ps_events = ps_events[ps_events.experiment == experiment]

    ps_events = correct_eegfile_field(ps_events)
    ps_events = attach_raw_bin_wrappers(ps_events)

    ps_sessions = np.unique(ps_events.session)
    print experiment, 'sessions:', ps_sessions

    compute_isi(ps_events)

    is_stim_event_type_vec = np.vectorize(is_stim_event_type)
    ps_events = ps_events[is_stim_event_type_vec(ps_events.type)]
    print len(ps_events), 'stim', experiment, 'events'

    ps_pow_mat_pre, ps_pow_mat_post = compute_ps_powers(ps_events, ps_sessions, channels, tal_info, experiment)

    joblib.dump(ps_pow_mat_pre, subject+'_ps_pow_mat_pre.pkl')
    joblib.dump(ps_pow_mat_post, subject+'_ps_pow_mat_post.pkl')

    prob_pre, prob_diff = compute_prob_deltas(ps_pow_mat_pre, ps_pow_mat_post, lr_classifier)

    joblib.dump(prob_pre, subject+'_prob_pre.pkl')
    joblib.dump(prob_diff, subject+'_prob_diff.pkl')

    print 'Fin'


if __name__ == "__main__":
    main()
