import os
import json
import zipfile
from os.path import *
from glob import glob
import shutil
import pathlib
from itertools import chain, cycle

import numpy as np
from sklearn.externals import joblib

from tornado.template import Template

from classiflib import ClassifierContainer, dtypes

from RamPipeline import *
from system_3_utils import ElectrodeConfigSystem3

CLASSIFIER_VERSION = '1.0.1'

class ExperimentConfigGeneratorClosedLoop5(RamTask):
    def __init__(self, params, mark_as_completed=False):
        RamTask.__init__(self, mark_as_completed)
        self.params = params

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
                file_abspath= os.path.join(root, file)
                file_abspath_segments = [x for x in pathlib.Path(str(file_abspath)).parts]

                # relative_path_segments = [x for x in  pathlib.Path(file_abspath).parts[root_path_len:]]
                # relative_path_segments.append(basename(file_abspath))

                relative_path = join(*file_abspath_segments[root_path_len:])


                ziph.write(os.path.join(root, file), relative_path )

    def run(self):
        class ConfigError(Exception):
            pass


        anodes = self.pipeline.args.anodes if self.pipeline.args.anodes else [self.pipeline.args.anode]
        cathodes = self.pipeline.args.cathodes if self.pipeline.args.cathodes else [self.pipeline.args.cathode]

        experiment = self.pipeline.args.experiment if self.pipeline.args.experiment else 'FR5'
        electrode_config_file = self.pipeline.args.electrode_config_file


        ec = ElectrodeConfigSystem3.ElectrodeConfig(electrode_config_file)
        config_chan_names =  [ec.stim_channels[stim_channel].name for stim_channel in ec.stim_channels]
        for stim_pair in zip(anodes,cathodes):
            if '_'.join(stim_pair) not in config_chan_names:
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
        stim_params_list = zip(anodes,cathodes,cycle(self.pipeline.args.min_amplitudes),
                               cycle(self.pipeline.args.max_amplitudes))
        for anode,cathode,min_amplitude,max_amplitude in stim_params_list:
            chan_label = '_'.join([anode,cathode])
            stim_params_dict[chan_label]={
                "min_stim_amplitude":min_amplitude,
                "max_stim_amplitude":max_amplitude,
                "stim_frequency":stim_frequency,
                "stim_duration":500,
                "stim_amplitude":stim_amplitude
            }

        fr5_stim_channel = '%s_%s'%(anodes[0],cathodes[0])
        project_dir_corename = 'experiment_config_dir/%s/%s'%(subject,experiment)
        project_dir = self.create_dir_in_workspace(project_dir_corename)

        # making sure we get the path even if the actual directory is not created
        project_dir = self.get_path_to_resource_in_workspace(project_dir_corename)

        config_files_dir = self.create_dir_in_workspace(abspath(join(project_dir,'config_files')))
        config_files_dir = self.get_path_to_resource_in_workspace(project_dir_corename+'/config_files')

        experiment_config_template_filename = join(dirname(__file__),'templates','{}_experiment_config.json.tpl'.format(
            'PS4_FR5' if 'PS4' in experiment else 'FR5'))
        experiment_config_template = Template(open(experiment_config_template_filename ,'r').read())

        electrode_config_file_core, ext = splitext(electrode_config_file)

        # FIXME: more intelligently set classifier filename
        experiment_config_content = experiment_config_template.generate(
            subject=subject,
            experiment=experiment,
            classifier_file='config_files/%s' % basename(classifier_path.replace('.pkl', '.zip')),
            classifier_version=CLASSIFIER_VERSION,
            stim_params_dict=stim_params_dict,
            electrode_config_file='config_files/%s' % basename(electrode_config_file_core+'.bin'),
            montage_file='config_files/%s' % basename(bipolar_pairs_path),
            excluded_montage_file='config_files/%s' % basename(excluded_pairs_path),
            biomarker_threshold=0.5,
            fr5_stim_channel=fr5_stim_channel,
            auc_all_electrodes = xval_full[-1].auc,
            auc_no_stim_adjacent_electrodes = xval_output[-1].auc,
        )

        experiment_config_file,experiment_config_full_filename = self.create_file_in_workspace_dir(project_dir_corename+'/experiment_config.json')
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
        pairs.sort(order='contact1')

        events = self.get_passed_object('FR_events')

        # FIXME: this is simplified from ComputeClassifier, but should really be more centralized instead of repeating
        sample_weight = np.ones(events.shape[0], dtype=np.float)
        sample_weight[events.type == 'WORD'] = self.params.encoding_samples_weight

        classifier = joblib.load(classifier_path)
        container = ClassifierContainer(
            classifier=classifier,
            pairs=pairs,
            features=joblib.load(self.get_path_to_resource_in_workspace(subject +
                                                                        ('' if self.pipeline.args.encoding_only else
                                                                        '-reduced_')
                                                                        +'pow_mat.pkl')),
            events=events,
            sample_weight=sample_weight,
            classifier_info={
                'auc': xval_output[-1].auc,
                'subject': subject
            }
        )
        container.save(
            join(config_files_dir, "{}-lr_classifier.zip".format(subject)),
            overwrite=True
        )

        # copying classifier pickle file
        # self.copy_pickle_resource_to_target_dir(classifier_path, config_files_dir)

        # copy pairs.json
        self.copy_resource_to_target_dir(bipolar_pairs_path,config_files_dir)

        # copy reduced_pairs.json
        self.copy_resource_to_target_dir(excluded_pairs_path,config_files_dir)



        self.copy_resource_to_target_dir(resource_filename=electrode_config_file_core+'.bin', target_dir=config_files_dir)
        self.copy_resource_to_target_dir(resource_filename=electrode_config_file_core + '.csv',
                                         target_dir=config_files_dir)
        #
        # # copy transformation matrix hdf5 file (if such exists)
        #
        # electrode_config_file_dir = dirname(electrode_config_file)
        # trans_matrix_fname =  join(electrode_config_file_dir,'monopolar_trans_matrix%s.h5'%subject)
        #
        # if self.pipeline.args.bipolar:
        #     if exists(trans_matrix_fname):
        #         self.copy_resource_to_target_dir(resource_filename=trans_matrix_fname,target_dir=config_files_dir)
        #     else:
        #         print ('. Could not find bipolar_2 monpopolar transformation matrix . You have requested bipolar referencing in the ENS ')
        #         print 'Configuration will be invalid'



        # zipping project_dir
        zip_filename = self.get_path_to_resource_in_workspace(
                        '%s_%s_'%(subject,experiment)
                        +'_'.join(chain(*[(anode,cathode,str(max_amplitude)) for anode,cathode,_,max_amplitude in stim_params_list]))
                        + '.zip')
        zipf = zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)
        self.zipdir(project_dir, zipf)
        zipf.close()

        print("Created experiment_config zip file: \n%s"%zip_filename)
