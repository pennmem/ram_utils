import os
import sys
import json
import pathlib
import warnings
import shutil
import zipfile
import numpy as np

from glob import glob
from itertools import cycle
from os.path import *
from sklearn.externals import joblib

from classiflib import ClassifierContainer, dtypes
from tornado.template import Template
from ramutils.pipeline import RamTask
from ramutils.log import get_logger
from ramutils.classifier.utils import get_pal_sample_weights


CLASSIFIER_VERSION = '1.0.1'
logger = get_logger()


class ExperimentConfigGeneratorClosedLoop5_V1(RamTask):
    def __init__(self, params, mark_as_completed=False):
        RamTask.__init__(self, mark_as_completed)
        self.params = params

    @property
    def classifier_container_path(self):
        classifier_path = self.get_passed_object(
            ('encoding_' if self.pipeline.args.encoding else '') +
            'classifier_path')
        if self.pipeline.args.classifier_type_to_output == 'pal':
            try:
                classifier_path = self.get_passed_object('classifier_path_pal')
            except KeyError:
                warnings.warn(
                    'Cannot generate PAL1-only classifier- most likely due to insufficient number of PAL1 sessions',
                    RuntimeWarning)
                sys.exit(1)
        classifier_container_path = basename(classifier_path).replace('.pkl',
                                                                      '.zip')
        return classifier_container_path

    def copy_resource_to_target_dir(self, resource_filename, target_dir):
        """
        Convenience fcn that copies file resurce into target directory
        :param resource_filename:
        :param target_dir:
        :return:
        """

        if isfile(resource_filename):
            resource_dst = join(target_dir, basename(resource_filename))
            try:

                shutil.copyfile(resource_filename, resource_dst)
            except shutil.Error as e:
                print ("Could not copy %s to %s . " % (resource_filename, resource_dst))
                pass  # ignore any copy errors

            return

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
            for file in files:
                file_abspath = os.path.join(root, file)
                file_abspath_segments = [x for x in pathlib.Path(str(file_abspath)).parts]
                relative_path = join(*file_abspath_segments[root_path_len:])
                ziph.write(os.path.join(root, file), relative_path)

    def run(self):
        anodes = self.pipeline.args.anodes if self.pipeline.args.anodes else [self.pipeline.args.anode]
        cathodes = self.pipeline.args.cathodes if self.pipeline.args.cathodes else [self.pipeline.args.cathode]
        experiment = self.pipeline.args.experiment if self.pipeline.args.experiment else 'PAL5'
        electrode_config_file = abspath(self.pipeline.args.electrode_config_file)

        events = self.get_passed_object('combined_evs')

        reduced_pow_mat = self.get_passed_object('reduced_pow_mat')
        if self.pipeline.args.encoding:
            reduced_pow_mat = self.get_passed_object('encoding_reduced_pow_mat')
            fr1_encoding_mask = (events.type == 'WORD') & (
            events.exp_name == 'FR1')
            pal1_encoding_mask = (events.type == 'WORD') & (
            events.exp_name == 'PAL1')
            encoding_mask = (fr1_encoding_mask | pal1_encoding_mask)
            events = events[encoding_mask]

        config_name = self.get_passed_object('config_name')
        subject = self.pipeline.subject.split('_')[0]
        stim_frequency = self.pipeline.args.pulse_frequency
        stim_amplitude = self.pipeline.args.target_amplitude
        bipolar_pairs_path = self.get_passed_object('bipolar_pairs_path')
        xval_output = self.get_passed_object('xval_output')
        excluded_pairs_path = self.get_passed_object('excluded_pairs_path')

        # Figure out which classifier should be used
        classifier_path = self.get_passed_object(('encoding_' if
                                                  self.pipeline.args.encoding
                                                  else '') + 'classifier_path')

        if self.pipeline.args.classifier_type_to_output == 'pal':
            try:
                classifier_path = self.get_passed_object('classifier_path_pal')
            except KeyError:
                warnings.warn(
                    'Cannot generate PAL1-only classifier- most likely due to insufficient number of PAL1 sessions',
                    RuntimeWarning)
                sys.exit(1)

        # TODO: Get this from the Odin config file instead?
        stim_params_dict = {}
        stim_params_list = zip(anodes, cathodes, cycle(self.pipeline.args.min_amplitudes),
                               cycle(self.pipeline.args.max_amplitudes))
        for anode, cathode, min_amplitude, max_amplitude in stim_params_list:
            chan_label = '_'.join([anode, cathode])
            stim_params_dict[chan_label] = {
                "min_stim_amplitude": min_amplitude,
                "max_stim_amplitude": max_amplitude,
                "stim_frequency": stim_frequency,
                "stim_duration": 500,
                "stim_amplitude": stim_amplitude
            }

        fr5_stim_channel = '%s_%s' % (anodes[0], cathodes[0])
        project_dir_corename = 'experiment_config_dir/%s/%s' % (subject, experiment)
        project_dir = self.create_dir_in_workspace(project_dir_corename)

        # making sure we get the path even if the actual directory is not created
        project_dir = self.get_path_to_resource_in_workspace(project_dir_corename)

        config_files_dir = self.create_dir_in_workspace(abspath(join(project_dir, 'config_files')))
        config_files_dir = self.get_path_to_resource_in_workspace(project_dir_corename + '/config_files')

        experiment_config_template_filename = join(dirname(__file__), 'templates',
                                                   '{}_experiment_config.json.tpl'.format(experiment))
        experiment_config_template = Template(open(experiment_config_template_filename, 'r').read())

        core_name_for_electrode_file = '{subject}_{config_name}'.format(subject=subject, config_name=config_name)

        if experiment == 'PAL5':
            experiment_config_content = experiment_config_template.generate(
                subject=subject,
                experiment=experiment,
                classifier_file='config_files/%s' % self.classifier_container_path,
                target_amplitude=self.pipeline.args.target_amplitude,
                electrode_config_file='config_files/{core_name_for_electrode_file}.bin'.format(
                    core_name_for_electrode_file=core_name_for_electrode_file),
                montage_file='config_files/%s' % basename(bipolar_pairs_path),
                excluded_montage_file='config_files/%s' % basename(excluded_pairs_path),
                biomarker_threshold=0.5,
                fr5_stim_channel=fr5_stim_channel
            )

        else:
            experiment_config_content = experiment_config_template.generate(
                subject=subject,
                experiment=experiment,
                classifier_file='config_files/%s' % self.classifier_container_path,
                stim_params_dict=stim_params_dict,
                electrode_config_file='config_files/{core_name_for_electrode_file}.bin'.format(
                    core_name_for_electrode_file=core_name_for_electrode_file),
                montage_file='config_files/%s' % basename(bipolar_pairs_path),
                excluded_montage_file='config_files/%s' % basename(excluded_pairs_path),
                biomarker_threshold=0.5,
                fr5_stim_channel=fr5_stim_channel
            )

        experiment_config_file, experiment_config_full_filename = self.create_file_in_workspace_dir(
            project_dir_corename + '/experiment_config.json')
        experiment_config_file.write(experiment_config_content)
        experiment_config_file.close()

        # Get all_pairs not from what is passed during the montage task, but the
        # actual set of all pairs
        all_pairs = self.get_passed_object('config_pairs_dict')
        all_pairs = all_pairs[subject]['pairs']

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


        # Sample weights are 1 if encoding classifier, otherwise it needs to
        # be calcualted
        sample_weights = get_pal_sample_weights(events,
                                                self.params.pal_samples_weight,
                                                self.params.encoding_samples_weight)
        if self.pipeline.args.encoding:
            sample_weights = np.ones(len(events))

        classifier = joblib.load(classifier_path)
        container = ClassifierContainer(
            classifier=classifier,
            pairs=pairs,
            features=reduced_pow_mat,
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
        self.copy_resource_to_target_dir(bipolar_pairs_path, config_files_dir)

        # copy reduced_pairs.json
        self.copy_resource_to_target_dir(excluded_pairs_path, config_files_dir)

        # vars for file moving/copy
        electrode_config_file_core, ext = splitext(electrode_config_file)
        electrode_config_file_dir = dirname(electrode_config_file)

        # renaming .csv file to the same core name as .bin file - see variable -  core_name_for_electrode_file
        old_csv_fname = electrode_config_file
        new_csv_fname = join(config_files_dir, core_name_for_electrode_file + '.csv')
        shutil.copy(old_csv_fname, new_csv_fname)

        try:
            old_bin_fname = join(electrode_config_file_dir, core_name_for_electrode_file + '.bin')
            new_bin_fname = join(config_files_dir, core_name_for_electrode_file + '.bin')
            shutil.copy(old_bin_fname, new_bin_fname)
        except IOError:
            raise IOError('Please make sure that binary electrode configuration file is '
                          'in the same directory as %s and is called %s' % (electrode_config_file, old_bin_fname))

        # zipping project_dir
        zip_filename = join(dirname(project_dir), experiment) + '.zip'
        zipf = zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)
        self.zipdir(project_dir, zipf)
        zipf.close()
