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


def compute_powers(events,monopolar_channels,bipolar_pairs,
                   start_time,end_time,buffer_time,
                   freqs,log_powers):

    if not isinstance(bipolar_pairs,np.recarray):
        bipolar_pairs = np.array(bipolar_pairs,dtype=[('ch0','S3'),('ch1','S3')]).view(np.recarray)
    sessions = np.unique(events.session)
    pow_mat = None
    for sess in sessions:
        sess_events = events[events.session==sess]
        # Load EEG
        eeg_reader = EEGReader(events=sess_events,channels=monopolar_channels,start_time=start_time,end_time=end_time)
        eeg = eeg_reader.read()
        if eeg_reader.removed_bad_data():
            print 'REMOVED SOME BAD EVENTS !!!'
            events = np.concatenate(events[events.session !=sess],eeg['events'].data.view(np.recarray))
            ev_order = np.argsort(events, order=('session', 'list', 'mstime'))
            events = events[ev_order]
            #The task will have to actually handle passing the new events
        eeg=eeg.add_mirror_buffer(duration=buffer_time)
        # Use bipolar pairs
        eeg= MonopolarToBipolarMapper(time_series=eeg,bipolar_pairs=bipolar_pairs).filter()
        #Butterworth filter to remove line noise
        eeg=ButterworthFilter(time_series=eeg,freq_range=[58.,62.],filt_type='stop').filter()

        try:
            sess_pow_mat=MorletWaveletFilterCpp(time_series=eeg,freqs = freqs,output='power').filter()
        except TypeError:
            print 'CPP Failed. Trying again.'
            sess_pow_mat,phase_mat = MorletWaveletFilter(time_series=eeg,freqs=freqs,output='power').filter()
        if log_powers:
            sess_pow_mat =  np.log10(sess_pow_mat)
        sess_pow_mat = np.nanmean(sess_pow_mat.transpose('events', 'bipolar_pairs', 'frequency', 'time'),-1)
        sess_pow_mat = zscore(sess_pow_mat)

        pow_mat = sess_pow_mat if pow_mat is None else np.concatenate((pow_mat,sess_pow_mat))

    pow_mat = pow_mat.reshape((len(events),len(bipolar_pairs)*len(freqs)))
    return pow_mat,events




