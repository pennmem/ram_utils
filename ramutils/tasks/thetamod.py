from typing import Optional

import numpy as np

from ramutils.tasks import task, make_task
from cmlreaders import CMLReader, get_data_index
from cmlreaders.eeg_container import EEGContainer

import ramutils.montage
import ramutils.events

from ramutils.thetamod import connectivity, tmi, artifact
from ramutils.exc import TooManySessionsError

__all__ = ['get_resting_connectivity',
           'get_psd_data',
           'compute_tmi']


@task(nout=4)
def get_psd_data(stim_events, rootdir):
    """

    Parameters
    ----------
    stim_events: np.rec.array
    rootdir: str

    Returns
    -------
    pre_psd:
        pre-stim power spectral density
    post_psd:
        post-stim power spectral density
    bad_events_mask:
        Boolean array indicating saturated events, as
        determined by analysis of post-stim EEG
    bad_channel_mask:
        Boolean array indicating bad channels,
    """
    subject, experiment, sessions = ramutils.events.extract_event_metadata(
        stim_events
    )
    reader = get_reader(subject,experiment,sessions[0],rootdir)

    pre_eeg = tmi.get_eeg('pre', reader, stim_events)
    post_eeg = tmi.get_eeg('post', reader, stim_events)

    pre_psd = tmi.compute_psd(pre_eeg)
    post_psd = tmi.compute_psd(post_eeg)
    bad_events_mask = artifact.get_bad_events_mask(post_eeg.data,
                                                    stim_events)

    bad_channel_mask = artifact.get_channel_exclusion_mask(pre_eeg.data,
                                                           post_eeg.data,
                                                           pre_eeg.samplerate)

    return pre_psd, post_psd, bad_events_mask, bad_channel_mask


@task()
def compute_tmi(stim_events: np.ndarray, pairs, \
                rootdir: str):

        subject = ramutils.events.extract_subject(stim_events)
        experiment = ramutils.events.extract_experiment_from_events(stim_events)
        sessions = ramutils.events.extract_sessions(stim_events)
        if len(sessions) != 1:
            raise TooManySessionsError

        session = sessions[0]

        reader = get_reader(subject, experiment, session, rootdir)
        stim_channels = make_task(tmi.get_stim_channels,
                                  pairs, stim_events)

        pre_eeg, post_eeg = [
            make_task(tmi.get_eeg, which, reader, stim_events, cache=False)
            for which in ("pre", "post")
        ]
        pre_psd = make_task(tmi.compute_psd, pre_eeg)
        post_psd = make_task(tmi.compute_psd, post_eeg)

        bad_events_mask = make_task(artifact.get_bad_events_mask, post_eeg.data,
                                    stim_events)

        distmat = make_task(ramutils.montage.get_distances, pairs)
        conn = make_task(get_resting_connectivity)

        channel_exclusion_mask = make_task(
            artifact.get_channel_exclusion_mask, pre_eeg.data,
            post_eeg.data, pre_eeg.samplerate)

        regressions, tstats = make_task(
            tmi.regress_distance,
            pre_psd, post_psd,
            conn, distmat, stim_channels,
            event_mask=bad_events_mask,
            artifact_channels=channel_exclusion_mask,
            nout=2)

        results = make_task(tmi.compute_tmi, regressions)

        return results


def get_reader(subject: Optional[str] = None,
               experiment: Optional[str] = None,
               session: Optional[int] = None, rootdir='/') -> CMLReader:
    """Return a reader for loading data. Defaults to the instance's subject,
    experiment, and session.

    """
    idx = get_data_index('r1', rootdir)

    montage = idx.loc[(idx.subject == subject)
                      & (idx.experiment == experiment)
                      & (idx.session == session)].montage.unique()[0]

    return CMLReader(subject, experiment, session,
                     montage=montage, rootdir=rootdir)


@task()
def get_resting_connectivity(subject, rootdir) -> np.ndarray:
    """Compute resting state connectivity."""
    df = get_data_index(rootdir=rootdir)

    # Read EEG data for "resting" events
    eeg_data = []

    # TODO: add repFR1 to the list of baseline experiments
    # TODO: make this contingent, see ASANA
    sessions = df.query("subject == @subject & experiment in ['FR1, catFR1', 'LocationSearch', 'RepFR1']")
    experiments = sessions.experiment.unique()

    if 'FR1' in experiments:
        sessions = sessions.query("experiment == 'FR1'")
    elif 'catFR1' in experiments:
        sessions = sessions.query("experiment == 'catFR1'")
    elif 'LocationSearch' in experiments:
        sessions = sessions.query("experiment == 'LocationSearch'")
    elif 'RepFR1' in experiments:
        sessions = sessions.query("experiment == 'RepFR1'")
    else:
        raise Exception("No experiments found for baseline connectivity")


    if len(sessions):
        for _, session in sessions.iterrows():
            # rootdir is omitted here, as the root
            # is already established

            reader = get_reader(subject=session.subject,
                                experiment=session.experiment,
                                session=session.session,
                                rootdir=rootdir)

            rate = reader.load('sources')['sample_rate']
            reref = not reader.load('sources')['name'].endswith('.h5')
            events = reader.load('task_events').query("type in ['COUNTDOWN_START', 'SHAM']")
            resting = connectivity.countdown_to_resting(events, rate)
            eeg = connectivity.read_eeg_data(reader, resting, reref=reref)
            eeg_data.append(eeg)

    eegs = EEGContainer.concatenate(eeg_data)
    conn = connectivity.get_resting_state_connectivity(eegs.to_mne(),
                                                       eegs.samplerate)
    return conn
