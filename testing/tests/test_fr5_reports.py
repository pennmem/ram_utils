"""
    Tests that the classifier output (weights and predicted probabilities) for
    the fr1 report, fr5 biomarker, and fr5 report all match the expected values
    based on MATLAB code. Note: These tests are only semi-automated. You will
    need to run the black-box functional tests in test_reports.py first for
    these tests to be using updated output.
"""
import os
import h5py
import pytest
import numpy as np

TEST_DIR = os.path.dirname(__file__)


@pytest.mark.parametrize("test_dir,subject", [
    ("/scratch/zduey/samplefr5_reports/{}/","R1308T"),
    ("/scratch/zduey/samplefr1_reports/{}/", "R1308T"),
    ("/scratch/zduey/sample_fr5biomarkers/{}/", "R1308T"),
    ("/scratch/zduey/samplefr5_reports/{}/","R1275D"),
    ("/scratch/zduey/samplefr1_reports/{}/", "R1275D"),
])
def test_compare_matlab_python_joint_classifier(test_dir, subject):
    """
        Compare the model coefficients and predicted probabilites from matlab and python
        implementations of the joint classifier. This is NOT an automated test. You must
        first run the necessary reports to ensure that the updated outputs exist in the
        noted directories
    """
    test_dir = test_dir.format(subject)
    matlab_file = h5py.File(TEST_DIR + "/test_data/{}_all.hdf5".format(subject), "r")
    coef_m = matlab_file['model_weights'][:].flatten()[:-1] # last term is intercept, so skip
    output_m = matlab_file['model_output'][:].flatten()

    python_file = h5py.File(test_dir + "{}-debug_data.h5".format(subject), "r")
    coef_p = python_file['model_weights'][:].flatten()
    output_p = python_file['model_output'][:].flatten()

    assert (len(coef_m) == len(coef_p))
    assert (len(output_m) == len(output_p))

    assert (np.allclose(coef_m, coef_p, atol=1e-2) == True)
    assert (np.allclose(output_m, output_p, atol=1e-2) == True)

    return
