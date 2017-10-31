import os
import zipfile
from os.path import *
import numpy as np
from scipy.io import savemat
import datetime
from subprocess import call
from tornado.template import Template
from glob import glob
import shutil
import pathlib
from itertools import cycle
from system_3_utils import ElectrodeConfigSystem3

from ramutils.pipeline import RamTask


class ExperimentConfigGeneratorClosedLoop5(RamTask):
    def __init__(self, params, mark_as_completed=False):
        RamTask.__init__(self, mark_as_completed)

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
        # try:
        #     # make_directory(full_dir_path=project_output_dir_tmp)
        #     mkdir_p(project_output_dir_tmp)
        # except IOError:

    def zipdir(self, path, ziph):

        root_paths_parts = pathlib.Path(str(path)).parts
        root_path_len = len(root_paths_parts)

        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                file_abspath = os.path.join(root, file)
                file_abspath_segments = [x for x in pathlib.Path(str(file_abspath)).parts]

                # relative_path_segments = [x for x in  pathlib.Path(file_abspath).parts[root_path_len:]]
                # relative_path_segments.append(basename(file_abspath))

                relative_path = join(*file_abspath_segments[root_path_len:])

                ziph.write(os.path.join(root, file), relative_path)

    def run(self):
        anodes = self.pipeline.args.anodes if self.pipeline.args.anodes else [self.pipeline.args.anode]
        cathodes = self.pipeline.args.cathodes if self.pipeline.args.cathodes else [self.pipeline.args.cathode]

        experiment = self.pipeline.args.experiment if self.pipeline.args.experiment else 'PAL5'
        electrode_config_file = abspath(self.pipeline.args.electrode_config_file)
        subject = self.pipeline.subject.split('_')[0]
        stim_frequency = self.pipeline.args.pulse_frequency
        stim_amplitude = self.pipeline.args.target_amplitude if 'PS4' not in self.pipeline.args.experiment else 'N/A'
        bipolar_pairs_path = self.get_passed_object('bipolar_pairs_path')
        classifier_path = (self.get_passed_object('encoding_classifier_path')
            if hasattr(self.pipeline.args, 'encoding') else self.get_passed_object('classifier_path'))
        excluded_pairs_path = self.get_passed_object('excluded_pairs_path')
        xval_full = self.get_passed_object('xval_output_all_electrodes')
        xval_output = self.get_passed_object('xval_output')

        retrieval_biomarker_threshold = self.get_passed_object('retrieval_biomarker_threshold')

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

        # making sure we get the path even if the actual directory is not created
        project_dir = self.get_path_to_resource_in_workspace(project_dir_corename)

        config_files_dir = self.get_path_to_resource_in_workspace(join(project_dir, 'config_files'))

        experiment_config_template_filename = join(dirname(__file__), 'templates',
                                                   '{}_experiment_config.json.tpl'.format(experiment))
        experiment_config_template = Template(open(experiment_config_template_filename, 'r').read())

        # vars for file moving/copy
        electrode_config_file_core, ext = splitext(basename(electrode_config_file))
        electrode_config_file_dir = dirname(electrode_config_file)

        experiment_config_content = experiment_config_template.generate(
            subject=subject,
            experiment=experiment,
            classifier_file='config_files/%s' % basename(classifier_path),
            stim_params_dict=stim_params_dict,

            electrode_config_file='config_files/{core_name_for_electrode_file}.bin'.format(
                core_name_for_electrode_file=electrode_config_file_core),

            montage_file='config_files/%s' % basename(bipolar_pairs_path),
            excluded_montage_file='config_files/%s' % basename(excluded_pairs_path),
            biomarker_threshold=0.5,
            retrieval_biomarker_threshold = retrieval_biomarker_threshold,
            fr5_stim_channel=fr5_stim_channel,
            auc_all_electrodes=xval_full[-1].auc,
            auc_no_stim_adjacent_electrodes=xval_output[-1].auc,
        )

        experiment_config_file, experiment_config_full_filename = self.create_file_in_workspace_dir(
            project_dir_corename + '/experiment_config.json')
        experiment_config_file.write(experiment_config_content)
        experiment_config_file.close()

        # copying classifier pickle file
        self.copy_pickle_resource_to_target_dir(classifier_path, config_files_dir)

        # copy pairs.json
        self.copy_resource_to_target_dir(bipolar_pairs_path, config_files_dir)

        # copy reduced_pairs.json
        self.copy_resource_to_target_dir(excluded_pairs_path, config_files_dir)

        try:
            old_bin_fname = join(electrode_config_file_dir, electrode_config_file_core + '.bin')
            new_bin_fname = join(config_files_dir, electrode_config_file_core+'.bin')
            shutil.copy(old_bin_fname, new_bin_fname)
        except IOError:
            raise IOError('Please make sure that binary electrode configuration file is '
                          'in the same directory as %s and is called %s' %(electrode_config_file,old_bin_fname) )



        # zipping project_dir
        zip_filename = '{subject}_{experiment}_{anode1}_{cathode1}_{maxamp1}_{anode2}_{cathode2}_{maxamp2}.zip'.format(
            subject=subject, experiment=experiment,
            anode1=anodes[0],cathode1=cathodes[0],maxamp1=self.pipeline.args.max_amplitudes[0],
            anode2=anodes[1],cathode2=cathodes[1],maxamp2=self.pipeline.args.max_amplitudes[1],
        )

        zipf = zipfile.ZipFile(self.get_path_to_resource_in_workspace(zip_filename), 'w', zipfile.ZIP_DEFLATED)
        self.zipdir(project_dir, zipf)
        zipf.close()
