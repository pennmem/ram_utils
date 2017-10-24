import numpy as np


class TercileStats(object):
    def __init__(self, probs, recalls):
        self.probs = probs
        self.recalls = recalls
        self.recall_rate = np.sum(recalls) / float(recalls.size)

    def run(self):
        self.thresh_low = np.percentile(self.probs, 100.0/3.0)
        thresh_high = np.percentile(self.probs, 2.0*100.0/3.0)

        low_terc_sel = (self.probs <= self.thresh_low)
        high_terc_sel = (self.probs >= thresh_high)
        mid_terc_sel = ~(low_terc_sel | high_terc_sel)

        low_terc_recall_rate = np.sum(self.recalls[low_terc_sel]) / float(np.sum(low_terc_sel))
        high_terc_recall_rate = np.sum(self.recalls[high_terc_sel]) / float(np.sum(high_terc_sel))
        mid_terc_recall_rate = np.sum(self.recalls[mid_terc_sel]) / float(np.sum(mid_terc_sel))

        self.low_pc_diff_from_mean = 100.0 * (low_terc_recall_rate-self.recall_rate) / self.recall_rate
        self.high_pc_diff_from_mean = 100.0 * (high_terc_recall_rate-self.recall_rate) / self.recall_rate
        self.mid_pc_diff_from_mean = 100.0 * (mid_terc_recall_rate-self.recall_rate) / self.recall_rate

        return self.low_pc_diff_from_mean, self.mid_pc_diff_from_mean, self.high_pc_diff_from_mean
