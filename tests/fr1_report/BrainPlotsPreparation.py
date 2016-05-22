__author__ = 'm'

from ptsa.data.readers import TalReader

from RamPipeline import *

from ReportUtils import ReportRamTask

from BrainPlotUtils import *

import numpy as np


class BrainPlotsPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=False):
        super(BrainPlotsPreparation,self).__init__(mark_as_completed)
        self.ttest_thresh = 0.01

    def extract_single_elec_tag(self,bipolar_ttest_data):
        tags_0 = map(lambda line_entry:line_entry[1].split('-')[0], bipolar_ttest_data)
        tags_1 = map(lambda line_entry:line_entry[1].split('-')[1], bipolar_ttest_data)

        s = set(tags_0+tags_1)

        return list(s)

    def find_monopolar_indices_for_elecs(self,tal_struct,elec_tags):
        monopol_tags = tal_struct.tagName

        et = np.array(elec_tags,dtype='|S32') # converting list of elec tags int numpy array

        index = np.argsort(monopol_tags)
        sorted_monopol_tags = monopol_tags[index]
        sorted_index = np.searchsorted(sorted_monopol_tags, et)
        # print sorted_index


        et_index = np.take(index, sorted_index, mode="clip")
        # mask = x[yindex] != y
        #
        # result = np.ma.array(yindex, mask=mask)
        # print result
        #
        matching_mask = monopol_tags[et_index] == et
        select_indices = et_index[matching_mask]

        return select_indices

    def run(self):
        cumulative_ttest_data_raw = self.get_passed_object('cumulative_ttest_data_raw')

        # tal_path = os.path.join('/Users/m','data/eeg',subject,'tal',subject+'_talLocs_database_monopol.mat')
        tal_path = os.path.join(self.pipeline.mount_point,'data/eeg',self.pipeline.subject,'tal',self.pipeline.subject+'_talLocs_database_monopol.mat')
        tal_reader = TalReader(filename=tal_path,struct_name='talStruct')
        tal_struct = tal_reader.read()

        pos_elecs_entries = [line for line in cumulative_ttest_data_raw if line [-1]>2.0]
        neg_elecs_entries = [line for line in cumulative_ttest_data_raw if line [-1]<-2.0]


        pos_elec_tags = self.extract_single_elec_tag(bipolar_ttest_data=pos_elecs_entries)
        neg_elec_tags = self.extract_single_elec_tag(bipolar_ttest_data=neg_elecs_entries)

        pos_monopol_indices = self.find_monopolar_indices_for_elecs(tal_struct,pos_elec_tags)

        neg_monopol_indices = self.find_monopolar_indices_for_elecs(tal_struct,neg_elec_tags)

        print pos_elec_tags
        print neg_elec_tags

        p = list(pos_monopol_indices) # pos elec indices
        n = list(neg_monopol_indices) # neg elec indices
        
        np_set = set(n+p) # set of pos and neg indices
        
        tot_set = set(list(range(tal_struct.shape[0])))  # set of all indices    
        
        non_sig_elec_indices =  list(tot_set-np_set) # non-significen elecs indices


        w = BrainPlotOffscreenWidget()
        w.set_size(1000,1000)


        lh = Hemisphere(hemi='l')
        rh = Hemisphere(hemi='r')

        w.add_display_object('lh',lh)
        w.add_display_object('rh',rh)


        # dg_elec = Electrodes(shape='sphere')
        
        pos_tal_struct = tal_struct[p]
        neg_tal_struct = tal_struct[n]        
        non_sig_tal_struct = tal_struct[non_sig_elec_indices]
        
        pos_tal_struct = pos_tal_struct[(pos_tal_struct.eType=='S') | (pos_tal_struct.eType=='G')]
        neg_tal_struct = neg_tal_struct[(neg_tal_struct.eType=='S') | (neg_tal_struct.eType=='G')]
        
        non_sig_tal_struct = non_sig_tal_struct[(non_sig_tal_struct.eType=='S') | (non_sig_tal_struct.eType=='G')]
        
        pos_elec = None
        neg_elec = None
        
        if pos_tal_struct.shape[0]:
            pos_elec_locs = np.vstack((pos_tal_struct.avgSurf.x,pos_tal_struct.avgSurf.y,pos_tal_struct.avgSurf.z)).T
            pos_elec = Electrodes(shape='sphere')
            pos_elec.set_electrodes_locations(loc_array=pos_elec_locs)
            pos_elec.set_electrodes_color(c=[255, 0, 0])
            w.add_display_object('pos_elec',pos_elec)


        if neg_tal_struct.shape[0]:
            neg_elec_locs = np.vstack((neg_tal_struct.avgSurf.x,neg_tal_struct.avgSurf.y,neg_tal_struct.avgSurf.z)).T
            neg_elec = Electrodes(shape='sphere')
            neg_elec.set_electrodes_locations(loc_array=neg_elec_locs)
            neg_elec.set_electrodes_color(c=[0, 0, 255])
            w.add_display_object('neg_elec',neg_elec)

        if non_sig_tal_struct.shape[0]:
            
            non_sig_elec_locs = np.vstack((non_sig_tal_struct.avgSurf.x,non_sig_tal_struct.avgSurf.y,non_sig_tal_struct.avgSurf.z)).T
            non_sig_elec = Electrodes(shape='sphere')
            non_sig_elec.set_electrodes_locations(loc_array=non_sig_elec_locs)
            non_sig_elec.set_electrodes_color(c=[128,128, 128])
            w.add_display_object('non_sig_elec',non_sig_elec)



        for camera_name in ('left','right','top','bottom'):

            w.load_camera_settings(filename=camera_name+'.camera.json')
            image_path = self.get_path_to_resource_in_workspace('reports','brain-'+camera_name+'.png')
            w.take_screenshot(filename=image_path)
            # w.take_screenshot(filename='brain-'+camera_name+'.png')




if __name__=='__main__':

    # algorithm to select indices of elements that are shared between two arrays
    import numpy as np
    x = np.array([3,5,7,1,9,8,6,6])
    y = np.array([2,1,5,10,100,6])

    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)


    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y

    result = np.ma.array(yindex, mask=mask)
    print result

    matching_mask = x[yindex] == y
    select_indices = yindex[matching_mask]

    print select_indices
