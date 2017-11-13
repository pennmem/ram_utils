import json
import hashlib
import os.path
import numpy as np
import pandas as pd

from sklearn.externals import joblib

from ReportUtils import RamTask
from ptsa.data.readers.IndexReader import JsonIndexReader


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

class MontagePreparation(RamTask):
    def __init__(self, params,mark_as_completed=True):
        super(MontagePreparation,self).__init__(mark_as_completed)
        self.params=params

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)
        for fname in bp_paths:
            with open(fname,'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def restore(self):
        subject = self.pipeline.subject

        monopolar_channels = joblib.load(self.get_path_to_resource_in_workspace(subject + '-monopolar_channels.pkl'))
        bipolar_pairs = joblib.load(self.get_path_to_resource_in_workspace(subject + '-bipolar_pairs.pkl'))
        bp_tal_structs = pd.read_pickle(self.get_path_to_resource_in_workspace(subject + '-bp_tal_structs.pkl'))
        bp_tal_stim_only_structs = pd.read_pickle(self.get_path_to_resource_in_workspace(subject + '-bp_tal_stim_only_structs.pkl'))
        reduced_pairs = joblib.load(self.get_path_to_resource_in_workspace(subject+'-reduced_pairs.pkl'))

        self.pass_object('reduced_pairs',reduced_pairs)
        self.pass_object('monopolar_channels', monopolar_channels)
        self.pass_object('bipolar_pairs', bipolar_pairs)
        self.pass_object('bp_tal_structs', bp_tal_structs)
        self.pass_object('bp_tal_stim_only_structs', bp_tal_stim_only_structs)

    def run(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])
        task = self.pipeline.task

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))
        bp_paths = json_reader.aggregate_values('pairs', subject=subj_code, montage=montage)

        bp_path = os.path.join(self.pipeline.mount_point, next(iter(bp_paths)))
        self.pass_object('bipolar_pairs_path', bp_path)

        f_pairs = open(bp_path, 'r')
        bipolar_data = json.load(f_pairs)[subject]['pairs']
        f_pairs.close()
        bipolar_data_stim_only = {bp_tag:bp_data for bp_tag,bp_data in bipolar_data.iteritems() if bp_data['is_stim_only']}
        bipolar_data = {bp_tag:bp_data for bp_tag,bp_data in bipolar_data.iteritems() if not bp_data['is_stim_only']}

        bp_tags = []
        bp_tal_structs = []
        for bp_tag,bp_data in bipolar_data.iteritems():
            ch1 = bp_data['channel_1']
            ch2 = bp_data['channel_2']
            bp_tags.append(bp_tag)
            bp_tal_structs.append(['%03d'%ch1, '%03d'%ch2, bp_data['type_1'], atlas_location(bp_data)])

        bp_tal_structs = pd.DataFrame(bp_tal_structs, index=bp_tags, columns=['channel_1', 'channel_2', 'etype', 'bp_atlas_loc'])
        bp_tal_structs.sort_values(by=['channel_1', 'channel_2'], inplace=True)

        monopolar_channels = np.unique(np.hstack((bp_tal_structs.channel_1.values,bp_tal_structs.channel_2.values)))
        bipolar_pairs = zip(bp_tal_structs.channel_1.values,bp_tal_structs.channel_2.values)

        bp_tal_stim_only_structs = pd.Series()
        if bipolar_data_stim_only:
            bp_tags_stim_only = []
            bp_tal_stim_only_structs = []
            for bp_tag,bp_data in bipolar_data_stim_only.iteritems():
                bp_tags_stim_only.append(bp_tag)
                bp_tal_stim_only_structs.append(atlas_location(bp_data))
            bp_tal_stim_only_structs = pd.Series(bp_tal_stim_only_structs, index=bp_tags_stim_only)
        try:
            events = self.get_passed_object('ps_events')
            stim_events = events[events.type=='STIM_ON']
            anode_nums = filter(None,np.unique(stim_events.anode_num))
            cathode_nums = filter(None,np.unique(stim_events.cathode_num))

        except Exception:
            events = self.get_passed_object('all_events')
            stim_events = events[events.type=='STIM_ON'].stim_params # returns a list of stim params
            anode_nums = filter(None,np.unique(stim_events.anode_number))
            cathode_nums = filter(None,np.unique(stim_events.cathode_number))

        stim_pairs = anode_nums+cathode_nums
        bipolar_pairs = np.array(bipolar_pairs, dtype=[('ch0', 'S3'), ('ch1', 'S3')]).view(np.recarray)
        include = [int(bp.ch0) not in stim_pairs and int(bp.ch1) not in stim_pairs for bp in bipolar_pairs]

        reduced_pairs = bipolar_pairs[np.array(include)]
        self.pass_object('reduced_pairs', reduced_pairs)
        joblib.dump(reduced_pairs, self.get_path_to_resource_in_workspace(subject + '-reduced_pairs.pkl'))

        self.pass_object('stim_pairs', stim_pairs)
        self.pass_object('monopolar_channels', monopolar_channels)
        self.pass_object('bipolar_pairs', bipolar_pairs)
        self.pass_object('bp_tal_structs', bp_tal_structs)
        self.pass_object('bp_tal_stim_only_structs', bp_tal_stim_only_structs)

        joblib.dump(monopolar_channels, self.get_path_to_resource_in_workspace(subject + '-monopolar_channels.pkl'))
        joblib.dump(bipolar_pairs, self.get_path_to_resource_in_workspace(subject + '-bipolar_pairs.pkl'))
        bp_tal_structs.to_pickle(self.get_path_to_resource_in_workspace(subject + '-bp_tal_structs.pkl'))
        bp_tal_stim_only_structs.to_pickle(self.get_path_to_resource_in_workspace(subject + '-bp_tal_stim_only_structs.pkl'))