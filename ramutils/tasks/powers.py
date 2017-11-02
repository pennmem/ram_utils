import numpy as np

from ptsa.data.readers import EEGReader
from ptsa.extensions.morlet.morlet import MorletWaveletTransform

from ramutils.log import get_logger
from ramutils.tasks import task

logger = get_logger()


@task()
def compute_powers(events, monopolar_channels, bipolar_pairs, params):
    """Computes powers for all events.

    :param np.recarray events: pre-filtered events for a single session
    :param int session:
    :param dict monopolar_channels:
    :param dict bipolar_pairs:
    :param ExperimentParameters params: experimental parameters
    :returns: power matrix
    :rtype: np.ndarray

    """
    n_freqs = len(params.freqs)
    n_bps = len(bipolar_pairs)
    n_events = len(events)
    sess_encoding_events_mask = (events.type == 'WORD')

    wavelet_transform = MorletWaveletTransform()
    wavelet_transform_retrieval = MorletWaveletTransform()

    eeg_reader = EEGReader(events=events, channels=monopolar_channels,
                           start_time=params.start_time,
                           end_time=params.end_time, buffer_time=0.0)

    eegs = eeg_reader.read().add_mirror_buffer(duration=params.buf)

    eeg_retrieval_reader = EEGReader(events=events, channels=monopolar_channels,
                                     start_time=params.retrieval_start_time,
                                     end_time=params.retrieval_end_time, buffer_time=0.0)

    eegs_retrieval = eeg_retrieval_reader.read().add_mirror_buffer(duration=params.retrieval_buf)

    samplerate = float(eegs.samplerate)

    # Wavelet transform encoding events
    winsize = int(round(samplerate*(params.end_time - params.start_time + 2*params.buf)))
    bufsize = int(round(samplerate*params.buf))
    logger.debug('samplerate = %f, winsize = %d bufsize =%d', samplerate, winsize, bufsize)
    pow_ev = np.empty(shape=n_freqs*winsize, dtype=float)
    wavelet_transform.init(params.width, params.freqs[0], params.freqs[-1], n_freqs, samplerate, winsize)

    # Wavelet transform retrieval events
    winsize_retrieval = int(round(samplerate*(params.retrieval_end_time - params.retrieval_start_time + 2*params.retrieval_buf)))
    bufsize_retrieval = int(round(samplerate*params.retrieval_buf))
    logger.debug('samplerate = %f, winsize_retrieval = %d, bufsize_retrieval = %d',
                 samplerate, winsize_retrieval, bufsize_retrieval)
    pow_ev_retrieval = np.empty(shape=n_freqs*winsize_retrieval, dtype=float)
    wavelet_transform_retrieval.init(params.width, params.freqs[0], params.freqs[-1], n_freqs, samplerate, winsize_retrieval)

    logger.info('Computing FR1/catFR1 powers')
    pow_mat = np.empty(shape=(n_events, n_bps, n_freqs), dtype=np.float)

    # FIXME: parallelize
    for i, bp in enumerate(bipolar_pairs):
        logger.info('Computing powers for bipolar pair %s', bp)
        elec1 = np.where(monopolar_channels == bp[0])[0][0]
        elec2 = np.where(monopolar_channels == bp[1])[0][0]

        bp_data = np.subtract(eegs[elec1], eegs[elec2])
        bp_data.attrs['samplerate'] = samplerate
        bp_data_retrieval = np.subtract(eegs_retrieval[elec1],eegs_retrieval[elec2])
        bp_data_retrieval.attrs['samplerate'] = samplerate
        bp_data = bp_data.filtered([58, 62], filt_type='stop', order=params.filt_order)
        bp_data_retrieval = bp_data_retrieval.filtered([58, 62], filt_type='stop', order=params.filt_order)

        n_enc = 0
        n_retr = 0

        for ev in range(n_events):
            if sess_encoding_events_mask[ev]:  # FIXME: ???
                wavelet_transform.multiphasevec(bp_data[n_enc][0:winsize], pow_ev)
                pow_ev_stripped = np.reshape(pow_ev, (n_freqs, winsize))[:,bufsize:winsize-bufsize]
                n_enc += 1
            else:
                wavelet_transform_retrieval.multiphasevec(bp_data_retrieval[n_retr][0:winsize_retrieval], pow_ev_retrieval)
                pow_ev_stripped = np.reshape(pow_ev_retrieval, (n_freqs, winsize_retrieval))[:,bufsize_retrieval:winsize_retrieval-bufsize_retrieval]
                n_retr += 1

            if params.log_powers:
                np.log10(pow_ev_stripped, out=pow_ev_stripped)

            pow_mat[ev, i, :] = np.nanmean(pow_ev_stripped, axis=1)

    return pow_mat

    # self.pow_mat = np.reshape(self.pow_mat, (len(events), n_bps*n_freqs))


@task()
def combine_session_powers(encoding_powers, retrieval_powers, events):
    """Combine power matrices computed for individual sessions.

    :param list encoding_powers:
    :param list retrieval_powers:
    :param np.recarray events:
    :returns: power matrix
    :rtype: np.ndarray

    """
    encoding_mat = np.concatenate(*encoding_powers)
    retrieval_mat = np.concatenate(*retrieval_powers)

    pow_mat = np.zeros((len(events), encoding_mat.shape[-1]))
    pow_mat[(events.type == 'WORD'), ...] = encoding_mat
    pow_mat[~(events.type == 'WORD'), ...] = retrieval_mat

    return pow_mat
