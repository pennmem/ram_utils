
import pytest
import functools
import numpy as np

from ramutils.controllability import assign_electrode_to_lausanne, \
modal_control, load_connectivity_matrix, calculate_modal_controllability
from pkg_resources import resource_filename
from scipy.io import loadmat

datafile = functools.partial(resource_filename,
                             'ramutils.test.test_data')


@pytest.mark.parametrize('subject, expected_failure', [
    ('invalid_mat', True),
    ('multiple_mat', True),
    ('no_directory', True),
    ('success', False)
])
def test_load_connectivity_matrix(subject, expected_failure):
    connectivity_matrix = load_connectivity_matrix(
        subject, rhino_root=datafile('/input/controllability/'))

    if expected_failure:
        assert connectivity_matrix is None

    else:
        assert connectivity_matrix is not None

    return


def test_calculate_modal_controllability():
    electrode_coordinates = np.load(
        datafile('/input/controllability/R1385E_electrode_coordinates.npy'))
    subject_connectivity = loadmat(datafile(
        '/input/controllability/R1385E_DTI_based_connectivity.mat'))
    subject_connectivity_mat = subject_connectivity['connectivity']

    actual_modal_control = calculate_modal_controllability(
        subject_connectivity_mat, electrode_coordinates)

    expected_modal_control = np.load(datafile(
        '/input/controllability/R1385E_modal_control_by_electrode.npy'))

    assert np.allclose(actual_modal_control, expected_modal_control)

    return


def test_assign_electrode_to_lausanne():
    electrode_coordinates = np.load(
        datafile('/input/controllability/R1385E_electrode_coordinates.npy'))
    expected_assignment = np.load(datafile(
        '/input/controllability/R1385E_electrode_roi_assignment.npy'))

    actual_assignment = assign_electrode_to_lausanne(electrode_coordinates)
    assert np.allclose(expected_assignment, actual_assignment)

    return


def test_modal_control():
    expected_connectivity_matrix = np.load(datafile(
        '/input/controllability/R1385E_modal_control_by_roi.npy'))

    subject_connectivity = loadmat(datafile(
        '/input/controllability/R1385E_DTI_based_connectivity.mat'))
    subject_connectivity_mat = subject_connectivity['connectivity']
    actual_connectivity_matrix = modal_control(subject_connectivity_mat)

    assert np.allclose(expected_connectivity_matrix,
                       actual_connectivity_matrix)

    return
