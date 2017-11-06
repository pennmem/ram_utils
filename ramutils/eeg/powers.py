""" Helper functions for computing powers from a set of EEG signals """

import time
import numpy as np

from ptsa.data.readers import EEGReader
from ptsa.data.filters import (
    MonopolarToBipolarMapper,
    MorletWaveletFilterCpp,
    MorletWaveletFilter,
    ButterworthFilter
)


def compute_powers(events, start_time, end_time, buffer_time, freqs,
                   log_powers, monopolar_channels=None, bipolar_pairs=None,
                   ComputePowers=None, filt_order=4, width=5):
    """ Compute powers using a Morlet wavelet filter

    Parameters
    ----------
    events: 
    start_time: 
    end_time: 
    buffer_time: 
    freqs: 
    log_powers: 
    monopolar_channels=None: 
    bipolar_pairs=None: 
    ComputePowers=None: 
    filt_order=4: 
    width=5: 

    Returns
    -------

    """
    # TODO: This should really be split up into a few smaller functions: load,
    # ButterworthFilter, WaveletFilter, etc.

    if (bipolar_pairs is not None) and (not isinstance(bipolar_pairs, np.recarray)):
        # it expects to receive a list
        bipolar_pairs = np.array(bipolar_pairs, dtype=[('ch0', 'S3'), ('ch1', 'S3')]).view(np.recarray)
    elif (bipolar_pairs is not None) and (isinstance(bipolar_pairs, np.recarray)):
        # to get the same treatment if we get recarray , we will convert it to a list and then back to
        # recarray with correct dtype
        bipolar_pairs = np.array(list(bipolar_pairs), dtype=[('ch0', 'S3'), ('ch1', 'S3')]).view(np.recarray)

    # since it's already not guaranteed that there will be a time series for each event
    n_events  = len(events)
    events = events[events['eegoffset'] >= 0]

    if n_events != len(events):
        print('Removed %s events with negative offsets'%(n_events-len(events)))

    sessions = np.unique(events.session)
    pow_mat = None
    tic = time.time()
    filter_time = 0.
    for sess in sessions:
        print('Loading EEG for session {}'.format(sess))
        sess_events = events[events.session == sess]
        # Load EEG
        if monopolar_channels is None:
            eeg_reader = EEGReader(events=sess_events,
                                   start_time=start_time,
                                   end_time=end_time)
        else:
            eeg_reader = EEGReader(events=sess_events,
                                   channels=monopolar_channels,
                                   start_time=start_time,
                                   end_time=end_time)

        try:
            eeg = eeg_reader.read()
        except IndexError: # recording was done in bipolar mode, and the channels are different than what we expect
            eeg_reader.channels = np.array([])
            eeg= eeg_reader.read() # FIXME: shouldn't we still be specifying everything except channels?

        samplerate = eeg['samplerate']
        if eeg_reader.removed_bad_data():
            print('REMOVED SOME BAD EVENTS !!!')
            events = np.concatenate((events[events.session != sess], 
                                     eeg['events'].data.view(np.recarray))).view(np.recarray)
            event_fields = events.dtype.names
            order = tuple(f for f in ['session', 'list', 'mstime'] if f in event_fields)
            ev_order = np.argsort(events, order=order)
            events = events[ev_order]
            # The task will have to actually handle passing the new events
        
        eeg = eeg.add_mirror_buffer(duration=buffer_time)

        # Use bipolar pairs if they exist and recording is not already bipolar
        if bipolar_pairs is not None and 'bipolar_pairs' not in eeg.coords:
            eeg = MonopolarToBipolarMapper(time_series=eeg, bipolar_pairs=bipolar_pairs).filter()
        elif 'bipolar_pairs' in eeg.coords and ComputePowers is not None:
            # FIXME: What class is this that is being passed and is that necessary?
            ComputePowers.bipolar_pairs = [('%03d'%a,'%03d'%b) for (a,b) in eeg['bipolar_pairs'].values]

        # Butterworth filter to remove line noise
        eeg = eeg.filtered(freq_range=[58., 62.], filt_type='stop', order=filt_order)
        eeg['samplerate'] = samplerate
        print('Computing powers')
        filter_tic = time.time()

        # making underlying array contiguous
        eeg.data = np.ascontiguousarray(eeg.data)
        wavelet_filter = MorletWaveletFilterCpp(time_series=eeg,
                                                freqs=freqs, 
                                                output='power',
                                                width=width,
                                                cpus=25)
        sess_pow_mat, phase_mat = wavelet_filter.filter()

        print('Total time for wavelet decomposition: %.5f s' % (time.time() - filter_tic))
        sess_pow_mat = sess_pow_mat.remove_buffer(buffer_time).data + np.finfo(np.float).eps/2.

        if log_powers:
            np.log10(sess_pow_mat, sess_pow_mat)
        sess_pow_mat = np.nanmean(sess_pow_mat.transpose(2, 1, 0, 3), -1)

        pow_mat = sess_pow_mat if pow_mat is None else np.concatenate((pow_mat, sess_pow_mat))


    pow_mat = pow_mat.reshape((len(events),-1))

    print('Total time elapsed: {}'.format(time.time() - tic))
    if ComputePowers is not None:
        ComputePowers.samplerate = eeg['samplerate'].data.astype(np.float)

    return pow_mat, events