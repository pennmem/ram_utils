import numpy as np
from math import isnan


class DeltaStats(object):
    def __init__(self, n_params, event_vals, param_grid, prob_pre, prob_diff, quantile):
        self.n_params = n_params
        self.event_vals = event_vals
        self.param_grid = param_grid
        self.prob_diff = prob_diff
        thresh_low = np.percentile(prob_pre, 100.0*quantile)
        thresh_high = np.percentile(prob_pre, 100.0*(1.0-quantile))
        self.low_quant_inds = (prob_pre<=thresh_low)
        self.high_quant_inds = (prob_pre>=thresh_high)

        self.mean_all = [None] * n_params
        self.mean_low = [None] * n_params
        self.mean_high = [None] * n_params

        self.stdev_all = [None] * n_params
        self.stdev_low = [None] * n_params
        self.stdev_high = [None] * n_params

    def run(self):
        for i in xrange(self.n_params):
            n = len(self.param_grid[i])

            self.mean_all[i] = np.empty(n)
            self.mean_low[i] = np.empty(n)
            self.mean_high[i] = np.empty(n)

            self.stdev_all[i] = np.empty(n)
            self.stdev_low[i] = np.empty(n)
            self.stdev_high[i] = np.empty(n)

            for j,param in enumerate(self.param_grid[i]):
                param_sel = (self.event_vals[i] == param)

                prob_diff_all = self.prob_diff[param_sel]
                mean = np.nanmean(prob_diff_all)
                if isnan(mean): mean = 0.0
                self.mean_all[i][j] = mean
                stdev = np.nanstd(prob_diff_all)
                if isnan(stdev): stdev = 0.0
                self.stdev_all[i][j] = stdev

                prob_diff_low = self.prob_diff[param_sel & self.low_quant_inds]
                mean = np.nanmean(prob_diff_low)
                if isnan(mean): mean = 0.0
                self.mean_low[i][j] = mean
                stdev = np.nanstd(prob_diff_low)
                if isnan(stdev): stdev = 0.0
                self.stdev_low[i][j] = stdev

                prob_diff_high = self.prob_diff[param_sel & self.high_quant_inds]
                mean = np.nanmean(prob_diff_high)
                if isnan(mean): mean = 0.0
                self.mean_high[i][j] = mean
                stdev = np.nanstd(prob_diff_high)
                if isnan(stdev): stdev = 0.0
                self.stdev_high[i][j] = stdev

    def y_range(self):
        y_min = y_max = 0.0
        for i in xrange(self.n_params):
            for j in xrange(len(self.param_grid[i])):
                y_min = min(y_min, self.mean_all[i][j]-self.stdev_all[i][j])
                y_min = min(y_min, self.mean_low[i][j]-self.stdev_low[i][j])
                y_min = min(y_min, self.mean_high[i][j]-self.stdev_high[i][j])

                y_max = max(y_max, self.mean_all[i][j]+self.stdev_all[i][j])
                y_max = max(y_max, self.mean_low[i][j]+self.stdev_low[i][j])
                y_max = max(y_max, self.mean_high[i][j]+self.stdev_high[i][j])

        return y_min, y_max
