import os
import json

import numpy as np
import pandas as pd

from sklearn.externals import joblib

from ReportUtils import ReportRamTask
from ptsa.data.readers import TalReader

from loc_to_df import loc_to_df

from glob import glob
from scipy.io import loadmat


def atlas_location(bp_data):
    atlases = bp_data['atlases']

    if 'stein' in atlases:
        loc_tag = atlases['stein']['region']
        if (loc_tag is not None) and (loc_tag!='') and (loc_tag!='None'):
            return loc_tag

    if (bp_data['type_1']=='D') and ('wb' in atlases):
        wb_loc = atlases['wb']['region']
        if (wb_loc is not None) and (wb_loc!='') and (wb_loc!='None'):
            return wb_loc

    if 'ind' in atlases:
        ind_loc = atlases['ind']['region']
        if (ind_loc is not None) and (ind_loc!='') and (ind_loc!='None'):
            return ('Left ' if atlases['ind']['x']<0.0 else 'Right ') + ind_loc

    return '--'

def atlas_location_matlab(tag, atlas_loc, comments):

    def colon_connect(s1, s2):
        if isinstance(s1, pd.Series):
            s1 = s1.values[0]
        if isinstance(s2, pd.Series):
            s2 = s2.values[0]
        return s1 if (s2 is None or s2=='' or s2 is np.nan) else s2 if (s1 is None or s1=='' or s1 is np.nan) else s1 + ': ' + s2

    if tag in atlas_loc.index:
        return colon_connect(atlas_loc.ix[tag], comments.ix[tag] if comments is not None else None)
    else:
        return '--'

class MontagePreparation(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(MontagePreparation,self).__init__(mark_as_completed)
        self.params = params
        self.bp_map = None

    def restore(self):
        subject = self.pipeline.subject

        monopolar_channels = joblib.load(self.get_path_to_resource_in_workspace(subject + '-monopolar_channels.pkl'))
        bipolar_pairs = joblib.load(self.get_path_to_resource_in_workspace(subject + '-bipolar_pairs.pkl'))
        bp_tal_structs = pd.read_pickle(self.get_path_to_resource_in_workspace(subject + '-bp_tal_structs.pkl'))

        self.pass_object('monopolar_channels', monopolar_channels)
        self.pass_object('bipolar_pairs', bipolar_pairs)
        self.pass_object('bp_tal_structs', bp_tal_structs)

    def run(self):
        subject = self.pipeline.subject

        try:
            raise ValueError()
            bp_path = os.path.join(self.pipeline.mount_point, 'protocols/r1/codes', subject, 'pairs.json')
            f_pairs = open(bp_path, 'r')
            bipolar_data = json.load(f_pairs)[subject]['pairs']
            f_pairs.close()
            bipolar_data = {bp_tag:bp_data for bp_tag,bp_data in bipolar_data.iteritems() if not bp_data['is_stim_only']}

            bp_tags = []
            bp_tal_structs = []
            for bp_tag,bp_data in bipolar_data.iteritems():
                bp_tags.append(bp_tag)
                ch1 = bp_data['channel_1']
                ch2 = bp_data['channel_2']
                bp_tal_structs.append(['%03d'%ch1, '%03d'%ch2, bp_data['type_1'], atlas_location(bp_data)])

            bp_tal_structs = pd.DataFrame(bp_tal_structs, index=bp_tags, columns=['channel_1', 'channel_2', 'etype', 'bp_atlas_loc'])
            bp_tal_structs.sort_values(by=['channel_1', 'channel_2'], inplace=True)
            monopolar_channels = np.unique(np.hstack((bp_tal_structs.channel_1.values,bp_tal_structs.channel_2.values)))
            bipolar_pairs = zip(bp_tal_structs.channel_1.values,bp_tal_structs.channel_2.values)

            bad_bps = np.zeros(len(bipolar_pairs), dtype=np.bool)
            if self.params.windex_cleanup:
                windex_files = glob(os.path.join(self.pipeline.mount_point,'home1/shennan.weiss/windex_out/%s-*_out.mat'%subject))
                for wfile in windex_files:
                    windex_out = loadmat(wfile, squeeze_me=True)['output_data'].item()
                    # lf_exclude
                    if windex_out[3] is not 0:
                        if isinstance(windex_out[3], (int,long)):
                            bad_bps[windex_out[3]-1] = True
                        else:
                            lf_exclude = windex_out[3].astype(int) - 1
                            bad_bps[lf_exclude] = True

                    # hf_exclude
                    if windex_out[7] is not 0:
                        if isinstance(windex_out[7], (int,long)):
                            bad_bps[windex_out[7]-1] = True
                        else:
                            hf_exclude = windex_out[7].astype(int) - 1
                            bad_bps[hf_exclude] = True
                print np.sum(bad_bps), 'bipolar pairs excluded:'
                bipolar_pairs = np.asarray(bipolar_pairs, dtype=np.dtype('O,O'))
                print bipolar_pairs[bad_bps]
                good_electrodes = ~bad_bps
                bp_tal_structs = bp_tal_structs[good_electrodes]
                bipolar_pairs = bipolar_pairs[good_electrodes]

            self.pass_object('monopolar_channels', monopolar_channels)
            self.pass_object('bipolar_pairs', bipolar_pairs)
            self.pass_object('bp_tal_structs', bp_tal_structs)

            joblib.dump(monopolar_channels, self.get_path_to_resource_in_workspace(subject + '-monopolar_channels.pkl'))
            joblib.dump(bipolar_pairs, self.get_path_to_resource_in_workspace(subject + '-bipolar_pairs.pkl'))
            bp_tal_structs.to_pickle(self.get_path_to_resource_in_workspace(subject + '-bp_tal_structs.pkl'))

        except:
            try:
                tal_path = os.path.join(self.pipeline.mount_point,'data/eeg',self.pipeline.subject,'tal',self.pipeline.subject+'_talLocs_database_bipol.mat')

                tal_reader = TalReader(filename=tal_path)

                monopolar_channels = tal_reader.get_monopolar_channels()
                bipolar_pairs = tal_reader.get_bipolar_pairs()

                matlab_bp_tal_struct = tal_reader.read()

                l_path = os.path.join(self.pipeline.mount_point, 'data/eeg', subject, 'docs/localization', subject+' Localization.xlsx')
                loc_info = loc_to_df(l_path)

                atlas_loc = None
                comments = None
                has_depth = ('Das Volumetric Atlas Location' in loc_info)
                has_surface_only = ('Freesurfer Desikan Killiany Surface Atlas Location' in loc_info)
                if has_depth or has_surface_only:
                    atlas_loc = loc_info['Das Volumetric Atlas Location' if has_depth else 'Freesurfer Desikan Killiany Surface Atlas Location']
                    comments = loc_info['Comments'] if ('Comments' in loc_info) else None

                bp_tags = []
                bp_tal_structs = []
                for i,bp in enumerate(matlab_bp_tal_struct):
                    tag = bp.tagName.upper().replace('_','\\textunderscore')
                    bp_tags.append(tag)
                    ch1 = bipolar_pairs[i][0]
                    ch2 = bipolar_pairs[i][1]
                    bp_atlas_loc = ''
                    try:
                        bp_atlas_loc = bp.locTag
                    except:
                        pass

                    if len(bp_atlas_loc)<4:
                        bp_atlas_loc = atlas_location_matlab(tag, atlas_loc, comments)

                    bp_tal_structs.append([ch1, ch2, bp.eType, bp_atlas_loc])

                bp_tal_structs = pd.DataFrame(bp_tal_structs, index=bp_tags, columns=['channel_1', 'channel_2', 'etype', 'bp_atlas_loc'])
                bp_tal_structs.sort_values(by=['channel_1', 'channel_2'], inplace=True)

                bad_bps = np.zeros(len(bipolar_pairs), dtype=np.bool)
                if self.params.windex_cleanup:
                    windex_files = glob(os.path.join(self.pipeline.mount_point,'home1/shennan.weiss/stas_0922/%s-*_out_r.mat'%subject))
                    for wfile in windex_files:
                        windex_out = loadmat(wfile, squeeze_me=True, variable_names=['hf_exclude','lf_exclude'])
                        # lf_exclude
                        # if windex_out[3] is not 0:
                        #     if isinstance(windex_out[3], (int,long)):
                        #         bad_bps[windex_out[3]-1] = True
                        #     else:
                        #         lf_exclude = windex_out[3].astype(int) - 1
                        #         bad_bps[lf_exclude] = True

                        # hf_exclude
                        hf_exclude = windex_out['hf_exclude']
                        if hf_exclude is not 0:
                            if isinstance(hf_exclude, (int,long)):
                                bad_bps[hf_exclude-1] = True
                            else:
                                hf_exclude = hf_exclude.astype(int) - 1
                                bad_bps[hf_exclude] = True
                    print np.sum(bad_bps), 'bipolar pairs excluded:'
                    print bipolar_pairs[bad_bps]
                    good_electrodes = ~bad_bps
                    self.bp_map = np.arange(len(bipolar_pairs))[good_electrodes]
                    bp_tal_structs = bp_tal_structs[good_electrodes]
                    bipolar_pairs = bipolar_pairs[good_electrodes]

                self.pass_object('monopolar_channels', monopolar_channels)
                self.pass_object('bipolar_pairs', bipolar_pairs)
                self.pass_object('bp_tal_structs', bp_tal_structs)
                self.pass_object('bp_map', self.bp_map)

                joblib.dump(monopolar_channels, self.get_path_to_resource_in_workspace(subject + '-monopolar_channels.pkl'))
                joblib.dump(bipolar_pairs, self.get_path_to_resource_in_workspace(subject + '-bipolar_pairs.pkl'))
                bp_tal_structs.to_pickle(self.get_path_to_resource_in_workspace(subject + '-bp_tal_structs.pkl'))

                bp_tal_structs.to_csv(self.get_path_to_resource_in_workspace(subject + '-bp_tal_structs.csv'), sep='\t')
            except:
                raise
