from RamPipeline import *

import TextTemplateUtils

import numpy as np
from scipy.io import savemat
import datetime
from subprocess import call


class SaveMatlabFile(RamTask):
    def __init__(self, params, mark_as_completed=False):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.biomarker_components = ['StatAccum.m', 'libfftw3-3*', 'morlet_interface*']

    def run(self):
        subject = self.pipeline.subject
        events = self.get_passed_object('events')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')

        n_bps = len(bipolar_pairs)
        n_chs = self.params.stim_params.n_channels

        bpmat = np.zeros(shape=(n_chs, n_bps), dtype=np.float)
        for i,bp in enumerate(bipolar_pairs):
            e1, e2 = bp[0],bp[1]
            bpmat[int(e1)-1, i] = 1
            bpmat[int(e2)-1, i] = -1

        lr_classifier = self.get_passed_object('lr_classifier')
        xval_output = self.get_passed_object('xval_output')[-1]
        probs = xval_output.probs
        thresh = xval_output.jstat_thresh

        mat_filename = '%s_%s_TH3_%s_%s_%dHz_%gmA.biomarker.mat' % (subject, datetime.date.today(), self.params.stim_params.anode, self.params.stim_params.cathode, self.params.stim_params.pulseFrequency, self.params.stim_params.amplitude/1000.0)

        mdict = {'Bio': {'Subject': subject,
                         'Version': self.params.version,
                         'Sessions': np.unique(events.session),
                         'W': lr_classifier.coef_[0],
                         'W0': lr_classifier.intercept_[0],
                         'trainingProb': probs[:,None],
                         'thresh': thresh,
                         'bpmat': bpmat,
                         'freqs': self.params.freqs,
                         'fs': self.get_passed_object('samplerate'),
                         'StimParams': {'elec1': self.params.stim_params.elec1,
                                        'elec2': self.params.stim_params.elec2,
                                        'amplitude': self.params.stim_params.amplitude,
                                        'duration': self.params.stim_params.duration,
                                        'trainFrequency': self.params.stim_params.trainFrequency,
                                        'trainCount' : self.params.stim_params.trainCount,
                                        'pulseFrequency': self.params.stim_params.pulseFrequency,
                                        'pulseCount': self.params.stim_params.pulseCount},
                         'filename': mat_filename}}

        mat_filename_in_workspace = self.get_path_to_resource_in_workspace(mat_filename)
        savemat(mat_filename_in_workspace, mdict)

        replace_dict = {
            'load FILL_IN': 'load ' + mat_filename
        }

        stim_control_in_workspace = self.get_path_to_resource_in_workspace('StimControl.m')
        TextTemplateUtils.replace_template(template_file_name='StimControl.m', out_file_name=stim_control_in_workspace, replace_dict=replace_dict)

        self.biomarker_components += [stim_control_in_workspace, mat_filename_in_workspace]
        zip_list = ' '.join(self.biomarker_components)

        biomarker_filename_in_workspace = mat_filename_in_workspace[:-4]  # cutoff .mat

        print 'Zipping biomarker file', biomarker_filename_in_workspace

        zip_command_str = 'zip -9 -j %s %s' % (biomarker_filename_in_workspace, zip_list)

        call([zip_command_str], shell=True)
