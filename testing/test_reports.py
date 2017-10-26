""" Functional tests for various reports """
import os
import pytest
import subprocess


# Update these before running
CODE_DIR = "/home1/zduey/ram_utils/"
MOUNT = "/"
WORKSPACE_BASE = "/scratch/zduey/"


@pytest.mark.parametrize("subject",[
    ("R1308T"),
    ("R1275D"),
]
)
def test_fr1_report(subject):
    os.chdir(CODE_DIR + "/tests/fr1_report/")
    print(os.getcwd())
    workspace = WORKSPACE_BASE + "samplefr1_reports/"
    command = "python fr1_report.py --subject={} --task=FR1 --workspace-dir={} --mount-point={}".format(subject, workspace, MOUNT)
    retcode = subprocess.call(command, shell=True)
    assert (retcode == 0)
    return

@pytest.mark.parametrize("subject",[
    ("R1308T"),
    ("R1275D"),
]
)
def test_fr5_report(subject):
    os.chdir(CODE_DIR + "/tests/fr5_report/")
    workspace = WORKSPACE_BASE + "samplefr5_reports/"
    command = "python fr5_report.py --subject={} --task=FR5 --workspace-dir={} --mount-point={}".format(subject, workspace, MOUNT)
    retcode = subprocess.call(command, shell=True)
    assert (retcode == 0)
    return

@pytest.mark.parametrize("subject, n_channels, anode, cathode, pulse_frequency, pulse_duration, target_amplitude, anode_num, cathode_num",[
    ("R1308T", "128", "LB6", "LB7", "200", "500", "250", "11", "12"),
]
)
def test_fr5_biomarker(subject, n_channels, anode, cathode, pulse_frequency, pulse_duration, target_amplitude, anode_num, cathode_num):
    os.chdir(CODE_DIR + "/tests/fr5_biomarker/")
    workspace = WORKSPACE_BASE + "samplefr5_biomarkers/"
    command = "python fr5_biomarker.py\
              --subject={}\
              --n-channels={}\
              --anode={}\
              --cathode={}\
              --pulse-frequency={}\
              --pulse-duration={}\
              --target-amplitude={}\
              --anode-num={}\
              --cathode-num={}\
              --workspace-dir={}\
              --mount-point={}".format(subject,
                                       n_channels,
                                       anode,
                                       cathode,
                                       pulse_frequency,
                                       pulse_duration,
                                       target_amplitude,
                                       anode_num,
                                       cathode_num,
                                       workspace,
                                       MOUNT)
    retcode = subprocess.call(command, shell=True)
    assert (retcode == 0)
    return

