import numpy as np
from ReportUtils import ReportRamTask
from scipy.stats import ttest_ind
from sklearn.externals import joblib


class ComputeTTest(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(ComputeTTest,self).__init__(mark_as_completed)
        self.params = params
        self.ttest = None
        self.selected_features = None

    def run(self):
        print 'Computing t-test per session and selecting features'

        subject = self.pipeline.subject
        task = self.pipeline.task

        ppc_features = self.get_passed_object('ppc_features')

        events = self.get_passed_object(task+'_events')
        sessions = np.unique(events.session)

        self.ttest = dict()
        for sess in sessions:
            sel = (events.session==sess)
            sess_events = events[sel]

            sess_ppc_features = ppc_features[sel,:]

            sess_recalls = np.array(sess_events.recalled, dtype=np.bool)

            recalled_sess_ppc_features = sess_ppc_features[sess_recalls,:]
            nonrecalled_sess_ppc_features = sess_ppc_features[~sess_recalls,:]

            t,p = ttest_ind(recalled_sess_ppc_features, nonrecalled_sess_ppc_features, axis=0)
            self.ttest[sess] = (t,p)

        n_features = ppc_features.shape[1]

        self.selected_features = dict()
        selected_features_all_sessions = np.ones(n_features, dtype=np.bool)
        for sess in sessions:
            sess_selected_features = np.ones(n_features, dtype=np.bool)
            for sess1 in sessions:
                if sess1!=sess:
                    sess_selected_features = sess_selected_features & (np.abs(self.ttest[sess1][0]) >= 2.0)
            print 'Session', sess, 'outsample test:', np.sum(sess_selected_features), 'out of', n_features, 'features selected'
            self.selected_features[sess] = sess_selected_features
            selected_features_all_sessions = selected_features_all_sessions & (np.abs(self.ttest[sess][0]) >= 2.0)
        print 'All sessions:', np.sum(selected_features_all_sessions), 'features selected'
        self.selected_features[-1] = selected_features_all_sessions

        self.pass_object('ttest', self.ttest)
        joblib.dump(self.ttest, self.get_path_to_resource_in_workspace(subject + '-' + task + '-ttest.pkl'))

        self.pass_object('selected_features', self.selected_features)
        joblib.dump(self.selected_features, self.get_path_to_resource_in_workspace(subject + '-' + task + '-selected_features.pkl'))

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        self.ttest = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-ttest.pkl'))
        self.pass_object('ttest', self.ttest)
        self.selected_features = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-selected_features.pkl'))
        self.pass_object('selected_features', self.selected_features)
