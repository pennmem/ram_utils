import warnings
from random import shuffle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ramutils.classifier import ModelOutput
from ramutils.classifier.utils import normalize_sessions, get_sample_weights
from ramutils.log import get_logger
from ramutils.tasks import task

logger = get_logger()


@task()
def compute_auc(classifier, features, recalls, mask):
    """Compute the AUC score.

    :param classifier:
    :param np.ndarray features:
    :param np.recarray recalls:
    :param mask:
    :returns: computed AUC value
    :rtype: float

    """
    masked_recalls = recalls[mask]
    probs = classifier.predict_proba(features[mask])[:, 1]
    auc = roc_auc_score(masked_recalls, probs)
    return auc


# FIXME: remove selfs
@task()
def run_lolo_xval(classifier, session, event_lists, recalls, permuted=False, sample_weights=None):
    """Run leave-one-session-out cross validation.

    :param LogisticRegression classifier:
    :param int session: session number
    :param ??? event_lists:
    :param ??? recalls:
    :param bool permuted:
    :param np.ndarray sample_weights:

    """
    probs = np.empty_like(recalls, dtype=np.float)
    lists = np.unique(event_lists)

    for lst in lists:
        insample_mask = (event_lists != lst)
        insample_pow_mat = self.pow_mat[insample_mask]
        insample_recalls = recalls[insample_mask]
        insample_samples_weights = sample_weights[insample_mask]

        # FIXME: what warnings are we ignoring and why?
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if sample_weights is not None:
                classifier.fit(insample_pow_mat, insample_recalls, insample_samples_weights)
            else:
                classifier.fit(insample_pow_mat, insample_recalls)

        outsample_mask = ~insample_mask
        outsample_pow_mat = self.pow_mat[outsample_mask]

        probs[outsample_mask] = classifier.predict_proba(outsample_pow_mat)[:, 1]

    if not permuted:
        xval_output = ModelOutput(recalls=recalls, probs=probs)
        xval_output.compute_roc()
        xval_output.compute_tercile_stats()
        self.xval_output[session] = self.xval_output[-1] = xval_output

    return probs


@task()
def compute_permuted_lolo_aucs(sess, event_lists, recalls, params,
                               sample_weights=None):
    """Compute permuted leave-one-list-out AUC scores.

    :param int sess: session number
    :param ??? event_lists:
    :param ??? recalls:
    :param ExperimentParameters params:
    :param np.ndarray sample_weights: sample weights

    """
    n_perm = params.n_perm
    permuted_recalls = np.array(recalls)
    AUCs = np.empty(shape=n_perm, dtype=np.float)
    for i in range(n_perm):
        for lst in event_lists:
            sel = (event_lists == lst)
            list_permuted_recalls = permuted_recalls[sel]
            shuffle(list_permuted_recalls)
            permuted_recalls[sel] = list_permuted_recalls
        probs = run_lolo_xval(sess, event_lists, permuted_recalls, permuted=True, sample_weights=sample_weights)
        AUCs[i] = roc_auc_score(recalls, probs)
        logger.info('AUC = %f', AUCs[i])
    return AUCs


# FIXME: remove selfs
# FIXME: split into smaller chunks
@task()
def compute_classifier(events, pow_mat, params):
    """Compute the classifier.

    :param str subject:
    :param np.recarray events:
    :param np.ndarray pow_mat:
    :param ExperimentParameters params:

    """
    encoding_mask = events.type == 'WORD'

    pow_mat[encoding_mask] = normalize_sessions(pow_mat[encoding_mask], events[encoding_mask])
    pow_mat[~encoding_mask] = normalize_sessions(pow_mat[~encoding_mask], events[~encoding_mask])

    classifier = LogisticRegression(C=params.C, penalty=params.penalty_type,
                                    solver='liblinear')

    event_sessions = events.session

    recalls = events.recalled
    recalls[events.type == 'REC_WORD'] = 1
    recalls[events.type == 'REC_BASE'] = 0

    sample_weights = get_sample_weights(events, params.encoding_samples_weight)

    sessions = np.unique(event_sessions)
    if len(sessions) > 1:
        logger.info('Performing permutation test')
        self.perm_AUCs = self.permuted_loso_AUCs(event_sessions, recalls, sample_weights,events=events)

        logger.info('Performing leave-one-session-out xval')
        _,encoding_probs = self.run_loso_xval(event_sessions, recalls, permuted=False,samples_weights=sample_weights, events=events)
        logger.info('CROSS VALIDATION ENCODING AUC = %f',
                    roc_auc_score(events[events.type == 'WORD'].recalled, encoding_probs))
    else:
        sess = sessions[0]
        event_lists = events.list

        logger.info('Performing in-session permutation test')
        self.perm_AUCs = self.permuted_lolo_AUCs(sess, event_lists, recalls,samples_weights=sample_weights)

        logger.info('Performing leave-one-list-out xval')
        self.run_lolo_xval(sess, event_lists, recalls, permuted=False,samples_weights=sample_weights)

    logger.info('CROSS VALIDATION AUC = %f', self.xval_output[-1].auc)

    pvalue = np.nansum(self.perm_AUCs >= self.xval_output[-1].auc) / float(self.perm_AUCs[~np.isnan(self.perm_AUCs)].size)
    logger.info('Perm test p-value = %f', pvalue)

    logger.info('thresh = %f, quantile = %f',
                self.xval_output[-1].jstat_thresh,
                self.xval_output[-1].jstat_quantile)

    classifier.fit(pow_mat, recalls, sample_weights)
    recall_prob_array = classifier.predict_proba(pow_mat)[:, 1]
    insample_auc = roc_auc_score(recalls, recall_prob_array)
    logger.info('in-sample AUC = %f', insample_auc)

    model_weights = classifier.coef_

    # FIXME
    # Specify that the file should overwrite so that when
    # ComputeClassifier and ComputeFullClassifier are run back to back,
    # it will not complain about the dataset already existing in the h5 file
    # try:
    #     self.save_array_to_hdf5(self.get_path_to_resource_in_workspace(subject + "-debug_data.h5"),
    #                             "model_output",
    #                             recall_prob_array,
    #                             overwrite=True)
    #     self.save_array_to_hdf5(self.get_path_to_resource_in_workspace(subject + "-debug_data.h5"),
    #                             "model_weights",
    #                         model_weights)
    # except Exception:
    #     print('could not save debug data')
