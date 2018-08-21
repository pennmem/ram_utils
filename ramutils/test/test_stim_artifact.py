from ramutils import stim_artifact
import ramutils.events
import ramutils.montage
import pytest
import ramutils.parameters


@pytest.mark.rhino
def test_get_tstats(rhino_root):
    subject, experiment, sessions = ('R1427T', 'TICL_FR', [0])
    paths = ramutils.parameters.FilePaths(root=rhino_root)
    events = ramutils.events.load_events(
        subject, experiment, sessions=sessions, rootdir=rhino_root
    )
    pairs = ramutils.montage.extract_pairs_dict(
        ramutils.montage.get_pairs(subject, experiment,
                                       sessions,
                                       paths)
    )
    stim_events = events[events.type == 'STIM_ON']
    assert sum(stim_events.list == -999) == 30

    tstats, pvals = stim_artifact.get_tstats(stim_events, pairs, return_pvalues=True)
    assert len(tstats) == len(pairs)
    assert sum(pvals < 0.001) == 140
