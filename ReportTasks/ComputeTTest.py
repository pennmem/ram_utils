from ReportUtils import ReportRamTask
import numpy as np
from scipy.stats.mstats import ttest_ind
from sklearn.externals import joblib

class ComputeTTest(ReportRamTask):
    def __init__(self,params,mark_as_completed=False):
        super(ReportRamTask,self).__init__(mark_as_completed)
        self.params=params

    @property
    def events(self):
        return self.get_passed_object('events')

    @property
    def pow_mat(self):
        return self.get_passed_object('pow_mat')

    def run(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        events = self.events
        sessions = np.unique(events.session)
        ttest = {}
        pow_mat = self.pow_mat

        for sess in sessions:
            sel = (events.session==sess)
            sess_events = events[sel]

            sess_pow_mat = pow_mat[sel,:]

            sess_recalls = np.array(sess_events.recalled, dtype=np.bool)

            recalled_sess_pow_mat = sess_pow_mat[sess_recalls,:]
            nonrecalled_sess_pow_mat = sess_pow_mat[~sess_recalls,:]

            t,p = ttest_ind(recalled_sess_pow_mat, nonrecalled_sess_pow_mat, axis=0)
            ttest[sess] = (t,p)

        recalls = np.array(events.recalled, dtype=np.bool)

        recalled_pow_mat = pow_mat[recalls,:]
        nonrecalled_pow_mat = pow_mat[~recalls,:]

        t,p = ttest_ind(recalled_pow_mat, nonrecalled_pow_mat, axis=0)
        #print t.shape
        #sys.exit(0)
        ttest[-1] = (t,p)

        self.pass_object('ttest', ttest)
        joblib.dump(self.ttest, self.get_path_to_resource_in_workspace(subject + '-' + task + '-ttest.pkl'))



