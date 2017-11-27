from __future__ import print_function

import os
import json
import zipfile
from os.path import *
from glob import glob
import shutil
import ramutils.pathlib
from itertools import chain, cycle

import numpy as np
from sklearn.externals import joblib

from tornado.template import Template

from classiflib import ClassifierContainer, dtypes
from bptools.odin import ElectrodeConfig

from ramutils.classifier.weighting import get_sample_weights
from ramutils.pipeline import *
from ramutils.log import get_logger

CLASSIFIER_VERSION = '1.0.1'
logger = get_logger()


class ExperimentConfigGeneratorClosedLoop5(RamTask):
    def __init__(self, params, mark_as_completed=False):
        RamTask.__init__(self, mark_as_completed)
        self.params = params

    @property
    def classifier_container_path(self):
        return basename(self.get_passed_object('classifier_path').replace('.pkl', '.zip'))

    def copy_resource_to_target_dir(self, resource_filename, target_dir):
        """
        Convenience fcn that copies file resurce into target directory
        :param resource_filename:
        :param target_dir:
        :return:
        """

        resource_dst = join(target_dir, basename(resource_filename))
        shutil.copyfile(resource_filename, resource_dst)

    def copy_pickle_resource_to_target_dir(self, resource_filename, target_dir):
        """
        Copies uncompressed pickle-ed resources -. Those resources are usually split into multiple files
        :param resource_filename: pickle resource
        :param target_dir: target directory
        :return: None
        """
        if isfile(resource_filename):
            pickled_files = glob(resource_filename + '*')
            for filename in pickled_files:
                self.copy_resource_to_target_dir(resource_filename=filename, target_dir=target_dir)

    def build_experiment_config_dir(self):
        pass

    def zipdir(self, path, ziph):
        root_paths_parts = pathlib.Path(str(path)).parts
        root_path_len = len(root_paths_parts)

        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for f in files:
                file_abspath = os.path.join(root, f)
                file_abspath_segments = [x for x in pathlib.Path(str(file_abspath)).parts]

                relative_path = join(*file_abspath_segments[root_path_len:])

                ziph.write(os.path.join(root, f), relative_path )

    def run(self):
        class ConfigError(Exception):
            pass

        anodes = self.pipeline.args.anodes if self.pipeline.args.anodes else [self.pipeline.args.anode]
        cathodes = self.pipeline.args.cathodes if self.pipeline.args.cathodes else [self.pipeline.args.cathode]

        experiment = self.pipeline.args.experiment if self.pipeline.args.experiment else 'FR5'
        electrode_config_file = self.pipeline.args.electrode_config_file

        # FIXME: these checks should have happened earlier with the other electrode conf checks
        ec = ElectrodeConfig(electrode_config_file)
        stim_channel_names = [ch.name for ch in ec.stim_channels]
        for stim_pair in zip(anodes, cathodes):
            if '_'.join(stim_pair) not in stim_channel_names:
                raise ConfigError('Stim channel %s is missing from electrode config file'%('_'.join(stim_pair)))

        subject = self.pipeline.subject.split('_')[0]
        stim_frequency = self.pipeline.args.pulse_frequency
        stim_amplitude = self.pipeline.args.target_amplitude
        bipolar_pairs_path = self.get_passed_object('bipolar_pairs_path')
        classifier_path = self.get_passed_object('classifier_path')
        excluded_pairs_path = self.get_passed_object('excluded_pairs_path')
        xval_full = self.get_passed_object('xval_output_all_electrodes')
        xval_output = self.get_passed_object('xval_output')

        stim_params_dict = {}
        stim_params_list = zip(anodes, cathodes,
                               cycle(self.pipeline.args.min_amplitudes),
                               cycle(self.pipeline.args.max_amplitudes))
        for idx, (anode, cathode, min_amplitude, max_amplitude) in enumerate(stim_params_list):
            chan_label = '_'.join([anode, cathode])
            stim_params_dict[chan_label] = {
                "min_stim_amplitude": min_amplitude,
                "max_stim_amplitude": max_amplitude,
                "stim_frequency": stim_frequency,
                "stim_duration": 500,
                "stim_amplitude": stim_amplitude[idx]
            }

        fr5_stim_channel = '%s_%s' % (anodes[0], cathodes[0])

        project_dir_corename = 'experiment_config_dir/%s/%s' % (subject, experiment)
        self.create_dir_in_workspace(project_dir_corename)
        project_dir = self.get_path_to_resource_in_workspace(project_dir_corename)

        self.create_dir_in_workspace(abspath(join(project_dir,'config_files')))
        config_files_dir = self.get_path_to_resource_in_workspace(project_dir_corename+'/config_files')

        if (experiment.lower() != "fr5") and (experiment.lower() != "catfr5"):
            # All experiments after FR5 share a similar config format to PS4,
            # namely the stim channels are defined in a dict-like form.
            prefix = 'PS4_FR5'
        else:
            prefix = 'FR5'
        template_filename = join(
            dirname(__file__), 'templates',
            '{}_experiment_config.json'.format(prefix)
        )

        with open(template_filename, 'r') as f:
            experiment_config_template = Template(f.read())

        electrode_config_file_core, ext = splitext(electrode_config_file)

        # FIXME: more intelligently set classifier filename
        experiment_config_content = experiment_config_template.generate(
            subject=subject,
            experiment=experiment,
            classifier_file='config_files/%s' % self.classifier_container_path,
            classifier_version=CLASSIFIER_VERSION,
            stim_params_dict=stim_params_dict,
            electrode_config_file='config_files/%s' % basename(electrode_config_file_core + '.bin'),
            montage_file='config_files/%s' % basename(bipolar_pairs_path),
            excluded_montage_file='config_files/%s' % basename(excluded_pairs_path),
            biomarker_threshold=0.5,
            fr5_stim_channel=fr5_stim_channel,
            auc_all_electrodes=xval_full[-1].auc,
            auc_no_stim_adjacent_electrodes=xval_output[-1].auc,
        )

        experiment_config_file, experiment_config_full_filename = self.create_file_in_workspace_dir(project_dir_corename+'/experiment_config.json')
        experiment_config_file.write(experiment_config_content)
        experiment_config_file.close()

        with open(bipolar_pairs_path, 'r') as f:
            all_pairs = json.load(f)[subject]['pairs']

        with open(excluded_pairs_path, 'r') as f:
            excluded_pairs = json.load(f)[subject]['pairs']

        used_pairs = {
            key: value for key, value in all_pairs.items()
            if key not in excluded_pairs
        }

        pairs = np.rec.fromrecords([
            (item['channel_1'], item['channel_2'],
             pair.split('-')[0], pair.split('-')[1])
            for pair, item in used_pairs.items()
        ], dtype=dtypes.pairs)
        pairs.sort(order='contact0')

        events = self.get_passed_object('FR_events')
        if self.pipeline.args.encoding_only:
            events = events[events.type=='WORD']

        # Re-compute sample weights for whatever events are included for config generation
        if events[events.type == 'REC_WORD'].shape[0] > 0:  # encoding-only classifier has no REC events
            sample_weights = get_sample_weights(events, self.params.encoding_samples_weight)
        else:  # encoding-only classifier
            sample_weights = np.ones(events.shape)

        classifier = joblib.load(classifier_path)
        container = ClassifierContainer(
            classifier=classifier,
            pairs=pairs,
            features=joblib.load(self.get_path_to_resource_in_workspace(subject +'-reduced_pow_mat.pkl')),
            events=events,
            sample_weight=sample_weights,
            classifier_info={
                'auc': xval_output[-1].auc,
                'subject': subject
            }
        )
        classifier_path = join(config_files_dir, self.classifier_container_path)
        logger.info('Saving classifier container to %s', classifier_path)
        container.save(classifier_path, overwrite=True)

        # copy pairs.json
        self.copy_resource_to_target_dir(bipolar_pairs_path,config_files_dir)

        # copy reduced_pairs.json
        self.copy_resource_to_target_dir(excluded_pairs_path,config_files_dir)

        self.copy_resource_to_target_dir(resource_filename=electrode_config_file_core+'.bin', target_dir=config_files_dir)
        self.copy_resource_to_target_dir(resource_filename=electrode_config_file_core + '.csv',
                                         target_dir=config_files_dir)

        # zipping project_dir
        zip_filename = self.get_path_to_resource_in_workspace(
                        '%s_%s_'%(subject,experiment)
                        +'_'.join(chain(*[(anode,cathode,str(max_amplitude)) for anode,cathode,_,max_amplitude in stim_params_list]))
                        + '.zip')
        zipf = zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)
        self.zipdir(project_dir, zipf)
        zipf.close()

        print("Created experiment_config zip file: \n%s"%zip_filename)
