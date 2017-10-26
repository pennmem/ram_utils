""" Functional tests for various reports """
import os
import shutil
import pytest
import subprocess


# Assumes that testing folder is in ramutils package and report code is in the tests/ folder at the top level
CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MOUNT = "/"
TEST_DIR = "/scratch/RAM_maint/automated_reports_json/tmp_testing_data/"

def setup_module():
    """ Called by pytest before any test functions are called """
    sample_reports = ["samplefr1_reports", "samplefr5_reports", "samplefr5_biomarkers"]
    for sample_report in sample_reports:
        tmpdir = TEST_DIR + sample_report
        if os.path.exists(tmpdir):
            shutil.rmtree(TEST_DIR + sample_report) # remove and replace
        os.mkdir(TEST_DIR + sample_report)
    return

@pytest.mark.parametrize("subject",[
    ("R1308T"),
    ("R1275D"),
]
)
def test_fr1_report(subject):
    os.chdir(CODE_DIR + "/tests/fr1_report/")
    print(os.getcwd())
    workspace = TEST_DIR + "samplefr1_reports/"
    command = "python fr1_report.py --subject={} --task=FR1 --workspace-dir={} --mount-point={}".format(subject, workspace, MOUNT)
    subprocess.check_output(command, shell=True)
    return

@pytest.mark.parametrize("subject",[
    ("R1308T"),
    ("R1275D"),
]
)
def test_fr5_report(subject):
    os.chdir(CODE_DIR + "/tests/fr5_report/")
    workspace = TEST_DIR + "samplefr5_reports/"
    command = "python fr5_report.py --subject={} --task=FR5 --workspace-dir={} --mount-point={}".format(subject, workspace, MOUNT)
    subprocess.check_output(command, shell=True)
    return

@pytest.mark.parametrize("subject, n_channels, anode, cathode, pulse_frequency, pulse_duration, target_amplitude, anode_num, cathode_num",[
    ("R1308T", "128", "LB6", "LB7", "200", "500", "250", "11", "12"),
]
)
def test_fr5_biomarker(subject, n_channels, anode, cathode, pulse_frequency, pulse_duration, target_amplitude, anode_num, cathode_num):
    os.chdir(CODE_DIR + "/tests/fr5_biomarker/")
    workspace = TEST_DIR + "samplefr5_biomarkers/"
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
    subprocess.check_output(command, shell=True)
    return

