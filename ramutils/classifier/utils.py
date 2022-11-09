""" Utility functions used during classifier training """

import os
import numpy as np

from glob import glob
from sklearn.linear_model import LogisticRegression

from classiflib.container import ClassifierContainer
from ramutils.exc import UnableToReloadClassifierException


def reload_classifier(subject, task, session, mount_point='/', base_path=None):
    """Loads the actual classifier used by Ramulator for a particular session

    Parameters
    ----------
    subject: str
        Subject ID
    task: str
        ex: FR5, FR6, PAL1, etc
    session: int
        Session number
    mount_point: str, default '/'
        Mount point for RHINO
    base_path: str
        Location of where the classifier files can be found. If None, default
        is to look in the expected location on RHINO

    Returns
    -------
    classifier_container: classiflib.container.ClassifierContainer

    """
    if base_path is None:
        base_path = os.path.join(mount_point, 'data', 'eeg', subject,
                                 'behavioral', task,
                                 'session_{}'.format(str(session)),
                                 'host_pc')

    config_paths = glob(os.path.join(base_path,'*', 'config_files'))
    if len(config_paths) < 1:
        # expected host_pc folder does not exist
        return None

    if len(config_paths) > 1:
        # Return the original classifier
        retrained_classifier_paths = glob(os.path.join(
            base_path, '*', 'config_files', 'retrained_classifier'))
        if len(retrained_classifier_paths):
            config_path = sorted(retrained_classifier_paths)[-1]
        else:
            config_path = sorted(config_paths)[-1]
        classifier_path = glob(os.path.join(config_path,
                                            '*classifier*.zip'))
        # No container was found, likely because it is the old .pkl version
        if len(classifier_path) == 0:
            return None
        classifier_path = classifier_path[0]
        orig_classifier_container = ClassifierContainer.load(classifier_path)
        return orig_classifier_container



    # FIXME: this needs a data quality check to confirm that all classifiers in
    # a session are the same!
    # We take the final timestamped directory because in principle retrained
    # classifiers can be different depending on artifact detection. In
    # reality, stim sessions should never be restarted (apart from issues
    # getting things started in the first place).
    config_path = config_paths[-1]
    if 'retrained_classifier' in os.listdir(config_path):
        classifier_path = glob(os.path.join(config_path,
                                            'retrained_classifier',
                                            '*classifier*.zip'))[0]
    else:
        classifier_path = glob(os.path.join(config_path,
                                            '*classifier*.zip'))
        if len(classifier_path) == 0:
            return None
        classifier_path = classifier_path[0]
    classifier_container = ClassifierContainer.load(classifier_path)

    return classifier_container


def train_classifier(pow_mat, events, sample_weights, penalty_param,
                     penalty_type, solver):
    """Train a classifier.

    Parameters
    ----------
    pow_mat : np.ndarray
    events : np.recarray
    sample_weights : np.ndarray
    penalty_param: Float
        Penalty parameter to use
    penalty_type: str
        Type of penalty to use for regularized model (ex: L2)
    solver: str
        Solver to use when fitting the model (ex: liblinear)

    Returns
    -------
    classifier : LogisticRegression
        Trained classifier

    """
    recalls = events.recalled
    classifier = LogisticRegression(C=penalty_param,
                                    penalty=penalty_type,
                                    solver=solver,
                                    class_weight='balanced')
    classifier.fit(pow_mat, recalls, sample_weights)
    return classifier


# TODO: group and average classifier weights by brain region
def save_classifier_weights_plot(weights, frequencies, pairs, file_):
    """
    Visualize the classifier weights as a function of frequency and location.

    Parameters
    -----------
    weights : np.ndarray (len(pairs)*len(frequencies)
    frequencies : np.ndarray[float]
    pairs : ??? Iterable describing the pairs in some way
    file_ : The destination of the classifier weight plot,
    which should be either a path or a file-like object.

    Returns:
    file_ : string or file-like
        The file_ parameter
    """
    # pyplot is imported here rather than at the top of the module to avoid any
    # problems setting the matplotlib backend elsewhere.
    from matplotlib import pyplot as plt

    weights_by_channel = weights.reshape((len(frequencies), len(pairs)))
    plt.imshow(weights_by_channel, aspect='auto', origin='lower', cmap='RdBu')
    cmin, cmax = weights.min(), weights.max()
    clim = max(abs(cmin), abs(cmax))
    plt.clim(-clim, clim)
    plt.colorbar()
    locs, old_labels = plt.yticks()
    new_labels = ['%d' % (np.rint(f).astype(int)) for f in frequencies]
    plt.yticks(locs[1:-1], new_labels)
    plt.savefig(file_,
                format="png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.1)
    plt.close()
    return file_
