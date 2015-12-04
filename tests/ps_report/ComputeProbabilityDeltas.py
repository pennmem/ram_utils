__author__ = 'm'


from RamPipeline import *

from sklearn.externals import joblib


class ComputeProbabilityDeltas(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params

    def restore(self):
        subject = self.pipeline.subject
        experiment = self.pipeline.experiment

        prob_pre = joblib.load(self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-prob_pre.pkl'))
        prob_diff = joblib.load(self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-prob_diff.pkl'))

        self.pass_object('prob_pre', prob_pre)
        self.pass_object('prob_diff', prob_diff)

    def run(self):
        subject = self.pipeline.subject
        experiment = self.pipeline.experiment

        lr_classifier = self.get_passed_object('lr_classifier')
        ps_pow_mat_pre = self.get_passed_object('ps_pow_mat_pre')
        ps_pow_mat_post = self.get_passed_object('ps_pow_mat_post')

        prob_pre, prob_diff = self.compute_prob_deltas(ps_pow_mat_pre, ps_pow_mat_post, lr_classifier)

        joblib.dump(prob_pre, self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-prob_pre.pkl'))
        joblib.dump(prob_diff, self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-prob_diff.pkl'))

        self.pass_object('prob_pre', prob_pre)
        self.pass_object('prob_diff', prob_diff)

    def compute_prob_deltas(self, ps_pow_mat_pre, ps_pow_mat_post, lr_classifier):
        prob_pre = lr_classifier.predict_proba(ps_pow_mat_pre)[:,1]
        prob_post = lr_classifier.predict_proba(ps_pow_mat_post)[:,1]
        return prob_pre, prob_post - prob_pre
