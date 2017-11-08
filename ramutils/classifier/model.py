import numpy as np
from traits.api import Float, Array
from sklearn.metrics import roc_curve, roc_auc_score

from ramutils.schema import Schema


class ModelOutput(Schema):
    """Statistics and scores for the output of the logistic regression
    classifier.

    In order to compute tercile stats and the ROC curve/AUC score, the
    :attr:`true_labels` and :attr:`probs` attributes must be set first.

    """
    true_labels = Array(desc='true binary labels')
    probs = Array(dtype=np.float64)
    auc = Float(dtype=np.float64, desc='AUC score')
    fpr = Array(dtype=np.float64, desc='false positive rate')
    tpr = Array(dtype=np.float64, desc='true positive rate')
    thresholds = Array(dtype=np.float64, desc='decreasing thresholds on the decision function')
    jstat_thresh = Float(desc='jstat threshold')
    jstat_quantile = Float(desc='jstat quantile')

    # TODO: descriptions
    low_pc_diff_from_mean = Float()
    mid_pc_diff_from_mean = Float()
    high_pc_diff_from_mean = Float()

    def _check_ready(self):
        if not len(self.true_labels) or not len(self.probs):
            raise RuntimeError("true_labels and probs must be set first")

    def compute_metrics(self):
        """Shorthand for calling both :meth:`compute_roc` and
        :meth:`compute_tercile_stats`.

        """
        self.compute_roc()
        self.compute_tercile_stats()

    def compute_roc(self):
        """Compute the ROC curve.

        :raises RuntimeError: when true_labels or probs aren't set yet

        """
        self._check_ready()

        try:
            self.auc = roc_auc_score(self.true_labels, self.probs)
        except ValueError:
            return

        self.fpr, self.tpr, self.thresholds = roc_curve(self.true_labels, self.probs)
        self.jstat_quantile = 0.5
        self.jstat_thresh = np.median(self.probs)

    def compute_tercile_stats(self):
        """Compute tercile stats.

        :raises RuntimeError: when true_labels or probs aren't set yet

        """
        self._check_ready()

        thresh_low = np.percentile(self.probs, 100.0 / 3.0)
        thresh_high = np.percentile(self.probs, 2.0 * 100.0 / 3.0)

        low_terc_sel = (self.probs <= thresh_low)
        high_terc_sel = (self.probs >= thresh_high)
        mid_terc_sel = ~(low_terc_sel | high_terc_sel)

        low_terc_recall_rate = np.sum(self.true_labels[low_terc_sel]) / float(np.sum(low_terc_sel))
        mid_terc_recall_rate = np.sum(self.true_labels[mid_terc_sel]) / float(np.sum(mid_terc_sel))
        high_terc_recall_rate = np.sum(self.true_labels[high_terc_sel]) / float(np.sum(high_terc_sel))

        recall_rate = np.sum(self.true_labels) / float(self.true_labels.size)

        self.low_pc_diff_from_mean = 100.0 * (low_terc_recall_rate - recall_rate) / recall_rate
        self.mid_pc_diff_from_mean = 100.0 * (mid_terc_recall_rate - recall_rate) / recall_rate
        self.high_pc_diff_from_mean = 100.0 * (high_terc_recall_rate - recall_rate) / recall_rate
