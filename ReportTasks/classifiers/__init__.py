from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np
from scipy.stats import zscore
from ..RamTaskMethods import compute_powers,ModelOutput

class RAMClassifier(object):
    """
    RAMClassifier is a container class for the various computations that define a RAM classification analysis from
    EEG and behavioral data.


    Relevant parameters are implemented as class attributes
    """
    C = 7.24e-4
    freqs = np.logspace(np.log10(6), np.log10(180))
    penalty = 'l2'
    solver = 'liblinear'
    class_weight = 'auto'

    start_time = 0.0
    end_time = 1.6
    buffer_time = 1.0

    def __init__(self,events,channels,pairs):
        """
        Construct the sklearn classifier, and
        :param events: Events for which to compute features
        :param channels: List of channel labels to use in the feature computation -- see `compute_features`
        :param pairs: Array of bipolar pairs to rereference the computation by -- see `compute_features`
"""
        self.lr_classifier = LogisticRegression(penalty=self.penalty,C=self.C,solver=self.solver,
                                                class_weight=self.class_weight)
        self.events = events
        self.channels = channels
        self.pairs = pairs
        self.pow_mat = np.zeros((len(events),len(pairs)*len(self.freqs)))

    @property
    def recalls(self):
        return self.events.recalled

    @property
    def sessions(self):
        return self.events.session

    @property
    def types(self):
        return self.events.type

    def compute_features(self):
        """
        By default, computes a power decomposition of the EEG signals corresponding to self.events, at frequencies
         specified by self.freqs
        As well as return `pow_mat` and `good_events`, `compute_features` also sets them as attributes of the RAMClassifier object
        :return: pow_mat
            The spectral decomposition of the desired EEG signals, zscored by session
        :return: good_events
            The events for which the EEG signal could successfully be read. Always a subset of `events`
        """
        pow_mat,good_events= compute_powers(self.events,self.channels,self.pairs,
                              self.start_time,self.end_time,self.buffer_time,
                              self.freqs,
                              log_powers=True)
        for session in np.unique(good_events.session):
            in_session = good_events.session == session
            pow_mat[in_session]= zscore(in_session,axis=0, ddof=1)
        self.pow_mat = pow_mat
        self.events = good_events
        return (pow_mat,good_events)



    def get_sample_weights(self,events):
        """
        Construct sample weights based on distribution of events. If implemented, the `class_weight` attribute should
        be `None`
        :param events: CML event structures
        :return: {np.array,None}
            An array of weights for each event
        """
        pass

    def fit(self,pow_mat,events):
        weights = self.get_sample_weights(events)
        self.lr_classifier.fit(pow_mat,self.recalls,sample_weight=weights)

    def predict_proba(self,pow_mat):
        """
        Returns the probability of success for an array of features. A light wrapper around the underlying classifier
        :param pow_mat:
        :return:
        """
        return self.lr_classifier.predict_proba(pow_mat)[:,1]

def cross_validate(ram_classifier, pow_mat):
    """
    Return cross_validated probabilities for the given classifier and power matrix, wrapped in a ModelOutput.

    :return:
    """
    events = ram_classifier.events
    sessions = np.unique(events.session)
    probs = np.zeros(len(events))
    for session in sessions:
        outsample = events.session == session
        insample = ~outsample
        ram_classifier.fit(pow_mat[insample], events[insample])
        probs[outsample] = ram_classifier.predict_proba(pow_mat[outsample])
    return ModelOutput(ram_classifier.recalls, probs)


def cross_validate_encoding_only(fr5_classifier, pow_mat):
    events = fr5_classifier.events
    sessions = np.unique(events.session)
    probs = np.array([np.nan] * len(events))
    for session in sessions:
        insample = events.session != session
        outsample = (~insample) & (events.type == 'WORD')
        fr5_classifier.fit(pow_mat[insample], events[insample])
        probs[outsample] = fr5_classifier.predict_proba(pow_mat[outsample])

    return ModelOutput(fr5_classifier.recalls[events.type == 'WORD'], probs[events.type == 'WORD'])


def permutation_pvalue(ram_classifier,parallel=None,n_iters=200):
    """
    Returns a p-value for the classifier's performance, based on a permutation test of the feature matrix.
    Note that after calling this function, the classifier will be fit to a random matrix, and should be refit before
    use.
    :param ram_classifier: a RAMClassifier object
    :param parallel: A joblib.Parallel object or None
    :return: np.int
        A p-value for the classifier
    """

    if parallel is None:
        aucs = np.empty(n_iters,dtype=float)
        for i in range(n_iters):
            output = cross_validate(ram_classifier,np.random.permutation(ram_classifier.pow_mat))
            output.compute_roc()
            print('AUC = %s'%output.auc)
            aucs[i] = output.auc
    else:
        outputs = parallel(joblib.delayed(cross_validate)(ram_classifier,np.random.permutation(ram_classifier.pow_mat))
                         for i in range(n_iters))
        for output in outputs:
            output.compute_roc()
        aucs = np.array([output.auc for output in outputs])

    true_output = cross_validate(ram_classifier,ram_classifier.pow_mat)
    true_output.compute_roc()
    return (aucs>=true_output.auc).sum()














