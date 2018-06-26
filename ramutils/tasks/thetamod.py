from typing import Optional

import numpy as np

from ramutils.tasks import task, make_task
from cmlreaders import CMLReader, get_data_index
from cmlreaders.timeseries import TimeSeries

import ramutils.montage
from ramutils.thetamod import connectivity, tmi, artifact

__all__ = ['compute_tmi']


@task()
def compute_tmi(subject: str, experiment: str, session: int,
                rootdir: str):

        reader = get_reader(subject, experiment, session, rootdir)
        pairs = reader.load("pairs").sort_values(by=["contact_1",
                                                     "contact_2"])

        stim_events = make_task(tmi.get_stim_events, reader)
        stim_channels = make_task(tmi.get_stim_channels,
                                  pairs, stim_events)

        pre_eeg, post_eeg = [
            make_task(tmi.get_eeg, which, reader, stim_events, cache=False)
            for which in ("pre", "post")
        ]

        bad_events_mask = make_task(artifact.get_bad_events_mask, post_eeg.data,
                                    stim_events)

        pre_psd = make_task(tmi.compute_psd, pre_eeg)
        post_psd = make_task(tmi.compute_psd, post_eeg)
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


def get_resting_connectivity(subject, rootdir) -> np.ndarray:
    """Compute resting state connectivity."""
    df = get_data_index(rootdir=rootdir)
    sessions = df[(df.subject == subject) &
                  (df.experiment == "FR1")].session.unique()

    if len(sessions) == 0:
        raise RuntimeError("No FR1 sessions exist for %s"%subject)
    # Read EEG data for "resting" events
    eeg_data = []
    for session in sessions:
        reader = get_reader(experiment="FR1", session=session)
        rate = reader.load('sources')['sample_rate']
        reref = not reader.load('sources')['name'].endswith('.h5')
        events = connectivity.get_countdown_events(reader)
        resting = connectivity.countdown_to_resting(events, rate)
        eeg = connectivity.read_eeg_data(reader, resting, reref=reref)
        eeg_data.append(eeg)

    eegs = TimeSeries.concatenate(eeg_data)
    conn = connectivity.get_resting_state_connectivity(eegs.to_mne(),
                                                       eegs.samplerate)
    return conn
