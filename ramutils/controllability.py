from __future__ import division

import os
import glob
import functools
import numpy as np

import scipy.io
import scipy.spatial.distance as dist
import scipy.linalg as scila

from ramutils.log import get_logger
from pkg_resources import resource_filename
datafile = functools.partial(resource_filename, 'ramutils.data')


logger = get_logger()


def load_connectivity_matrix(subject, rhino_root="/"):
    """ Load the DTI-based subject-specific connectivity matrix

    Parameters
    ----------
    subject: str
        Subect identifier used to find the connectivity matrix

    root: str
        Mount point for RHINO

    Returns
    -------
    connectivity_matrix: np.ndarray or None
        If connectivity matrix is found,
    """
    location = "data/eeg/{}/controllability/".format(subject)
    expected_path = os.path.join(rhino_root, location)

    if os.path.exists(expected_path) is False:
        logger.error('Controllability folder not found in /data/eeg/[SUBJECT]/')
        return

    connectivity_matrices = glob.glob(expected_path + "*.mat")
    if len(connectivity_matrices) != 1:
        logger.error("Multiple connectivity matrices found, there must only "
                     "be one")
        return

    connectivity_matrix_data = scipy.io.loadmat(connectivity_matrices[0])
    if 'connectivity' not in connectivity_matrix_data.keys():
        logger.error("MATLAB file does not contain 'connectivity' key")
        return

    connectivity_matrix = connectivity_matrix_data['connectivity']

    return connectivity_matrix


def calculate_modal_controllability(A, electrode_mni_coords):
    """ Computes a modal controllability value for each electrode

    Parameters
    ----------
    A: np.ndarray
        R x R matrix of structural connections corresponding to anatomical
        brain regions parcellated using the Lausanne atlas.


    electrode_mni_coords: np.ndarray
        N x 3 matrix of three-dimensional (x,y,z) coordinates the N ECoG
        electrodes in MNI space.

    Returns
    --------
    values -- np.array
        N x 1 vector of modal controllability values corresponding to each of
        the N electrodes.

    Notes
    -----
    Bassett Lab, University of Pennsylvania, 2017.

    """

    # 463 x 1 vector of lausanne atlas locations for given coordinates
    elec_roi_assign = assign_electrode_to_lausanne(electrode_mni_coords)

    # 463 x 1 vector of controllability for each region
    modal_val_roi = modal_control(A)
    modal_val_elec = modal_val_roi[elec_roi_assign]

    return modal_val_elec


def assign_electrode_to_lausanne(electrode_mni_coords):
    """
        This function assigns electrodes to the Lausanne anatomical atlas
        (463 brain regions) based on the MNI-coordinates of the electrodes.
        Specifically, an electrode is assigned to one of the Lausanne brain
        regions based on the regional assignment of its nearest neighboring
        voxel.

    Parameters
    ----------
        electrode_mni_coords: np.ndarray
            N x 3 matrix of three-dimensional (x,y,z) coordinates for each
            electrode in a subject's montage in MNI space

    Returns
    -------
        values: np.array
            N x 1 vector of integers corresponding to one of 463 brain regions
            for the N electrodes

    Notes
    -----
    Provided by the Bassett Lab, University of Pennsylvania, 2017.

    """

    lausanne_atlas = np.load(datafile("ROIv_scale250_dilated_nii_voxels.npy"))
    lausanne_voxels = lausanne_atlas[:, :3]
    lausanne_roi_id = lausanne_atlas[:, 3]

    # Get data properties
    [n_chan, n_dims] = np.shape(electrode_mni_coords)

    # Locate closest voxel to an electrode and assign electrode
    # to that voxel's associated brain region
    dist_to_voxels = dist.cdist(electrode_mni_coords, lausanne_voxels)
    values = lausanne_roi_id[np.argmin(dist_to_voxels, axis=1)]

    return values


def modal_control(A):
    """
    Stimulation of brain regions with high modal controllability was
    positively associated with average change in memory classifier output
    (deliverable: 3.2.1.4(c)). Therefore, it is recommended to compute modal
    controllability using this function, and stimulate the brain region with
    the highest modal controllability value. Returns values of MODAL
    CONTROLLABILITY for each node in a network, given the adjacency matrix
    for that network. Modal controllability indicates the ability of that
    node to steer the system into difficult-to-reach states, given input at
    that node.

    Parameters
    ----------
    A: np.ndarray
        The structural (NOT FUNCTIONAL) network adjacency matrix,
        such that the simple linear model of dynamics outlined in the
        reference is an accurate estimate of brain state fluctuations.
        Assumes all values in the matrix are positive, and that the
        matrix is symmetric.

    Returns
    -------
    phi: np.array
        Vector of modal controllability values for each node

    Notes:
    ------
    Bassett Lab, University of Pennsylvania, 2016.
    Reference: Gu, Pasqualetti, Cieslak, Telesford, Yu, Kahn, Medaglia,
               Vettel, Miller, Grafton & Bassett, Nature Communications 6:8414,
               2015.
    """
    # Normalize the matrix based on largest singular value
    A = A / (1 + np.linalg.svd(A)[1][0])

    # Evaluate schur stability
    T, U = scila.schur(A, 'real')
    eigVals = np.diag(T);

    N = A.shape[0]
    phi = np.zeros(N)
    for ii in range(N):
        phi[ii] = np.dot(U[ii, :]**2, 1 - eigVals**2)

    return phi
