import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def compute_roc_metrics(predicted_probs, true_labels):
    """
        Generate basic metrics related to classifier evaluation including
        AUC, false positive rate, true positive rate, median predicted
        probability

    Parameters
    ----------
    predicted_probs: array_like
        List of predicted probabilities
    true_labels: array_like
        List of expected binary outcome labels

    Returns
    -------
    dict
        Classifier evaluation metrics

    """
    auc = roc_auc_score(true_labels, predicted_probs)

    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    jstat_quantile = 0.5
    jstat_thresh = np.median(predicted_probs)

    metrics = {
        'auc': auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'jstat_quantile': jstat_quantile,
        'jstat_thres': jstat_thresh
    }
    return metrics


def compute_tercile_metrics(predicted_probs, true_labels):
    """ Compute summary metrics comparing recall rates for each tercile of
    classifier output with the overall recall rate

    Parameters
    ----------
    predicted_probs: array_like
        List of predicted probabilities
    true_labels: array_like
        List of expected binary outcome lablese

    Returns
    -------
    dict
        Percent difference between recall rate at each tercile of classifier
        output with the overall recall rate

    Notes
    -----
    This is primarily used for plots in the reports

    """
    thresh_low = np.percentile(predicted_probs, 100.0 / 3.0)
    thresh_high = np.percentile(predicted_probs, 2.0 * 100.0 / 3.0)

    low_tercile_mask = (predicted_probs <= thresh_low)
    high_tercile_mask = (predicted_probs >= thresh_high)
    mid_tercile_mask = ~(low_tercile_mask | high_tercile_mask)

    low_terc_recall_rate = np.sum(true_labels[low_tercile_mask]) / float(np.sum(
        low_tercile_mask))
    mid_terc_recall_rate = np.sum(true_labels[mid_tercile_mask]) / float(np.sum(
        mid_tercile_mask))
    high_terc_recall_rate = np.sum(true_labels[high_tercile_mask]) / float(
        np.sum(high_tercile_mask))

    recall_rate = np.sum(true_labels) / float(true_labels.size)

    low_pc_diff_from_mean = 100.0 * (low_terc_recall_rate - recall_rate) / recall_rate
    mid_pc_diff_from_mean = 100.0 * (mid_terc_recall_rate - recall_rate) / recall_rate
    high_pc_diff_from_mean = 100.0 * (high_terc_recall_rate - recall_rate) / recall_rate

    results = {
        'low_pc_diff_from_mean': low_pc_diff_from_mean,
        'mid_pc_diff_from_mean': mid_pc_diff_from_mean,
        'high_pc_diff_from_mean': high_pc_diff_from_mean
    }

    return results
