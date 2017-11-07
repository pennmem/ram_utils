""" Functional tests for various reports """
import os
import shutil
import pytest
import subprocess
import classiflib


# Assumes that testing folder is in ramutils package and report code is in the tests/ folder at the top level
CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MOUNT = "/"
TEST_DIR = "/scratch/RAM_maint/automated_reports_json/tmp_testing_data/"
TEST_DATA = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/test_data/"

def cleanup():
    """ Utility function for setting up test directory. Call manually before
        running a new batch of tests
    """
    sample_reports = ["samplefr1_reports", "samplefr5_reports",
                      "samplefr5_biomarkers", "samplefr6_reports",
                      "samplepal1_reports",
                      "samplepal5_reports", "samplepal5_biomarkers",
                      "samplethr1_reports", "samplethr3_reports"]
    for sample_report in sample_reports:
        if os.path.exists(TEST_DIR + sample_report):
           shutil.rmtree(TEST_DIR + sample_report)
        os.mkdir(TEST_DIR + sample_report)
    return

@pytest.mark.parametrize("subject",[
    ("R1308T"),
    ("R1275D"),
]
)
def test_fr1_report(subject):
    os.chdir(CODE_DIR + "/tests/fr1_report/")
    workspace = TEST_DIR + "samplefr1_reports/"
    command = "python fr1_report.py --subject={} --task=FR1 --workspace-dir={} --mount-point={}".format(subject, workspace, MOUNT)
    subprocess.check_output(command, shell=True)
    assert os.path.exists(workspace + "{}/reports/{}_FR1_report.pdf".format(subject, subject))
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
    assert os.path.exists(workspace + "{}/reports/{}-FR5_report.pdf".format(subject, subject))
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


@pytest.mark.parametrize("subject,experiment,electrode_config_file,pulse_frequency,target_amplitude,anode, cathode,min_amplitudes,max_amplitudes",[
        ("R1350D", "FR5", TEST_DATA + "R1350D_18OCT2017L0M0NOSTIM.csv",
         "200", "1.0", "LAD6", "LAD7", "0.1", "0.5"),
]
)
def test_fr5_util_system_3(subject, experiment, electrode_config_file,
                           pulse_frequency, target_amplitude, anode,
                           cathode, min_amplitudes, max_amplitudes):
    os.chdir(CODE_DIR + "/tests/fr5_biomarker/system3/")
    workspace = TEST_DIR + "samplefr5_biomarkers/"
    command = "python fr5_util_system_3.py\
              --subject={}\
              --experiment={}\
              --electrode-config-file={}\
              --anode={}\
              --cathode={}\
              --pulse-frequency={}\
              --target-amplitude={}\
              --min-amplitudes={}\
              --max-amplitudes={}\
              --workspace-dir={}\
              --mount-point={}".format(subject,
                                       experiment,
                                       electrode_config_file,
                                       anode,
                                       cathode,
                                       pulse_frequency,
                                       target_amplitude,
                                       min_amplitudes,
                                       max_amplitudes,
                                       workspace,
                                       MOUNT)
    subprocess.check_output(command, shell=True)
    output_classifier = (workspace +
                         "experiment_config_dir/{}/{}/config_files/{}-lr_classifier.zip"
                        ).format(subject, experiment, subject)
    classifier = classiflib.ClassifierContainer.load(output_classifier)
    # If the sample weighting failed, there will only be 1s and 2.5s
    assert (min(classifier.sample_weight) < 1)
    return



@pytest.mark.parametrize("subject",[
    ("R1333N"),
]
)
def test_pal1_report(subject):
    os.chdir(CODE_DIR + "/tests/pal1_report/")
    workspace = TEST_DIR + "samplepal1_reports/"
    command = "python pal1_report.py\
                --subject={}\
                --task=PAL1\
                --workspace-dir={}\
                --mount-point={}".format(subject, workspace, MOUNT)
    subprocess.check_output(command, shell=True)
    assert os.path.exists(workspace + "{}/reports/{}_PAL1_report.pdf".format(subject, subject))
    return


@pytest.mark.parametrize("subject, classifier",[
    ("R1312N", "pal"),
]
)
def test_pal5_report(subject, classifier):
    os.chdir(CODE_DIR + "/tests/pal5_report/")
    workspace = TEST_DIR + "samplepal5_reports/"
    command = "python pal5_report.py\
                --subject={}\
                --task=PAL5\
                --classifier={}\
                --workspace-dir={}\
                --mount-point={}".format(subject, classifier, workspace, MOUNT)
    subprocess.check_output(command, shell=True)
    assert os.path.exists(workspace + "{}/reports/PAL5-{}-report.pdf".format(subject, subject))
    return


def test_pal5_biomarker():
    return


@pytest.mark.parametrize("subject,experiment,electrode_config_file,pulse_frequency,min_amplitudes,max_amplitudes,anodes,cathodes",[
         ("R1333N", "PAL5", TEST_DATA + "R1333N_28AUG2017L0M0STIM.csv",
          "200", ["0.25","0.25"], ["1.0", "1.0"], ["LPLT5", "LAHD21"], ["LPLT6", "LAHD22"]),
 ]
 )
def test_pal5_util_system_3(subject, experiment, electrode_config_file, 
                            pulse_frequency, min_amplitudes, max_amplitudes, 
                            anodes, cathodes):
    os.chdir(CODE_DIR + "/tests/pal5_biomarker/system3/")
    workspace = TEST_DIR + "samplepal5_biomarkers/"
    command = "python pal5_util_system_3.py\
               --subject={}\
               --experiment={}\
               --electrode-config-file={}\
               --anodes={}\
               --cathodes={}\
               --pulse-frequency={}\
               --min-amplitudes={}\
               --max-amplitudes={}\
               --workspace-dir={}\
               --mount-point={}".format(subject,
                                        experiment,
                                        electrode_config_file,
                                        anodes,
                                        cathodes,
                                        pulse_frequency,
                                        min_amplitudes,
                                        max_amplitudes,
                                        workspace,
                                        MOUNT)
    #subprocess.check_output(command, shell=True)
    return


# PAL5 has a different argument parsing scheme that prevents us from testing the
# way that the fr5 util can be tested
def test_pal5_util_system_3():
    os.chdir(CODE_DIR + "/tests/pal5_biomarker/system3/")
    command = "python pal5_util_system_3.py"
    #subprocess.check_output(command, shell=True)
    return


@pytest.mark.parametrize("subject",[
    ("R1328E"),
]
)
def test_thr1_report(subject):
    os.chdir(CODE_DIR + "/tests/thr1_report/")
    workspace = TEST_DIR + "samplethr1_reports/"
    command = "python thr1_report.py\
               --subject={}\
               --task=THR1\
               --workspace-dir={}\
               --mount-point={}".format(subject, workspace, MOUNT)
    subprocess.check_output(command, shell=True)
    assert os.path.exists(workspace + "{}/reports/{}_THR1_report.pdf".format(subject, subject))
    return

def test_thr3_report():
    return


@pytest.mark.parametrize("subject",[
    ("R1342M"),
]
)
def test_joint_fr1_catfr1_report(subject):
    os.chdir(CODE_DIR + "/tests/fr1_catfr1_joint_report/")
    workspace = TEST_DIR + "sample_fr1_catfr1_joint_reports/"
    command = "python fr1_catfr1_joint_report.py\
               --subject={}\
               --task=FR1\
               --workspace-dir={}\
               --mount-point={}".format(subject, workspace, MOUNT)
    subprocess.check_output(command, shell=True)
    assert os.path.exists(workspace + "{}/reports/{}_FR1_catFR1_joined_report.pdf ".format(subject, subject))
    return


@pytest.mark.parametrize("subject",[
    ("R1293P"),
]
)
def test_fr_stim_report(subject):
    os.chdir(CODE_DIR + "/tests/fr_stim_report/")
    workspace = TEST_DIR + "sample_fr_stim_reports/"
    command = "python fr_stim_report.py\
               --subject={}\
               --task=FR3\
               --workspace-dir={}\
               --mount-point={}".format(subject, workspace, MOUNT)
    subprocess.check_output(command, shell=True)
    assert os.path.exists(workspace + "{}/reports/FR3-{}-report.pdf".format(subject, subject))
    return

# Note: The special mount-point for this test is temporary until we have actual
# FR6 data to use.
@pytest.mark.parametrize("subject, mount_point",[
    ("R1350D","/scratch/zduey/testing/fr6_mock/"),
]
)
def test_fr6_report(subject, mount_point):
    os.chdir(CODE_DIR + "/tests/fr6_report/")
    workspace = TEST_DIR + "catfr6_reports/"
    command = "python fr6_report.py\
               --subject={}\
               --task=catFR6\
               --workspace-dir={}\
               --mount-point={}".format(subject, workspace, mount_point)
    subprocess.check_output(command, shell=True)
    assert os.path.exists(workspace + "{}/reports/{}-catFR6_report.pdf".format(subject, subject))
    return