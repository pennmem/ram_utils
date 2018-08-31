import ramutils.powers
import scipy.stats
import cmlreaders
import json
import pandas as pd


def get_tstats(stim_events, pairs, start_time, duration, return_pvalues=False, before_experiment=True,):
    """
    Computes ttest on the average EEG value pre-stim vs post-stim.
    TODO: import from artdet; define parameters centrally

    Parameters
    ----------
    stim_events: np.rec.array
      Stimulation events for a session
    pairs: dict
      bipolar pairs
    start_time: int
      time after stim offset/before stim onset to begin (seconds)
    duration: int
      Length of eeg to evaluate (seconds)
    return_pvalues: bool
      If `true`, return p-values along with t-statistics
    before_experiment:
      If `true`, only include stim events before the first list
    Returns
    -------
    t: np.ndarray
      T-statistics by channel
    p: np.ndarray
      p-values by channel
    """

    stim_duration = 0.500

    if before_experiment:
        # Only use stim events from artifact detection period
        stim_events = stim_events[stim_events['list'] == -999]
        if len(stim_events) < 30: #TODO: MAKE THIS A CONFIG PARAMETER -- SEE TICL_FR expconf generator
            return (None, None) if return_pvalues else None

    pre_stim_eeg = ramutils.powers.load_eeg(
        stim_events,
        start_time=-(start_time+duration),
        end_time=-start_time,
        bipolar_pairs=pairs
    )
    post_stim_eeg = ramutils.powers.load_eeg(
        stim_events,
        start_time=stim_duration+start_time,
        end_time=stim_duration+start_time+duration,
        bipolar_pairs=pairs
    )

    means = [interval.mean(-1) for interval in [post_stim_eeg, pre_stim_eeg]]
    t, p = scipy.stats.ttest_rel(*means, axis=1)
    if return_pvalues:
        return t, p
    else:
        return t


def get_artifact_detection_info(subject,experiment,session, paths):
    """
    Loads artifact detection information from Ramulator event log

    Parameters
    ----------
    subject
    experiment
    session
    paths

    Returns
    -------
    artifact_info: dict

    Notes
    -----
    Requires cmlreaders
    """

    finder = cmlreaders.PathFinder(subject, experiment, session,
                                   rootdir=paths.root)
    with open(finder.find('event_log')) as event_log_file:
        event_log = json.load(event_log_file)
    event_df = pd.DataFrame.from_records(event_log['events'])
    artifact_rows = event_df.loc[event_df.event_label.apply(
        lambda x: x.startswith('ARTIFACT_DETECTION'))].set_index('event_label')
    artifact_rows.index = [x.remove('ARTIFACT_DETECTION_').lower()
                           for x in artifact_rows.index]
    return artifact_rows.T.loc['msg_stub'].to_dict()
