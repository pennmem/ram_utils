import json
import urllib
import urllib2

import numpy as np
import pandas as pd

from sklearn.externals import joblib

from ReportUtils import ReportRamTask


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
        ind_loc = ('Left ' if atlases['ind']['x']<0.0 else 'Right ') + atlases['ind']['region']
        if (ind_loc is not None) and (ind_loc!='') and (ind_loc!='None'):
            return ind_loc

    return '--'


class MontagePreparation(ReportRamTask):
    def __init__(self, params, mark_as_completed=True):
        super(MontagePreparation,self).__init__(mark_as_completed)
        self.params = params

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
            url_params = urllib.urlencode({'codes':subject, 'atlases':'stein,wb,ind'})

            req = urllib2.Request(self.params.api_bipolar_url+'?'+url_params)
            f = urllib2.urlopen(req)
            bipolar_data = json.load(f)[subject]['pairs']
            f.close()

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

            self.pass_object('monopolar_channels', monopolar_channels)
            self.pass_object('bipolar_pairs', bipolar_pairs)
            self.pass_object('bp_tal_structs', bp_tal_structs)

            joblib.dump(monopolar_channels, self.get_path_to_resource_in_workspace(subject + '-monopolar_channels.pkl'))
            joblib.dump(bipolar_pairs, self.get_path_to_resource_in_workspace(subject + '-bipolar_pairs.pkl'))
            bp_tal_structs.to_pickle(self.get_path_to_resource_in_workspace(subject + '-bp_tal_structs.pkl'))

        except:
            raise
            # self.raise_and_log_report_exception(
            #                                     exception_type='MissingDataError',
            #                                     exception_message='Missing or corrupt electrodes data for subject %s' % subject
            #                                    )
