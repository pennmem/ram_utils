__author__ = 'm'

from ptsa.data.readers import TalReader

from RamPipeline import *

from ReportUtils import ReportRamTask

from BrainPlotUtils import *

import numpy as np


class BrainPlotsPreparation(ReportRamTask):
    def __init__(self, mark_as_completed=False):
        super(BrainPlotsPreparation, self).__init__(mark_as_completed)
        self.ttest_thresh = 0.01
        self.pos_color = [255, 0, 0]
        self.neg_color = [0, 0, 255]
        self.non_sig_color = [128, 128, 128]

    def extract_single_elec_tag(self, bipolar_ttest_data):
        tags_0 = map(lambda line_entry: line_entry[1].split('-')[0], bipolar_ttest_data)
        tags_1 = map(lambda line_entry: line_entry[1].split('-')[1], bipolar_ttest_data)

        s = set(tags_0 + tags_1)

        return list(s)

    def find_monopolar_indices_for_elecs(self, tal_struct, elec_tags):
        monopol_tags = tal_struct.tagName

        et = np.array(elec_tags, dtype='|S32')  # converting list of elec tags int numpy array

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
        tal_path = os.path.join(self.pipeline.mount_point, 'data/eeg', self.pipeline.subject, 'tal',
                                self.pipeline.subject + '_talLocs_database_monopol.mat')
        tal_reader = TalReader(filename=tal_path, struct_name='talStruct')
        tal_struct = tal_reader.read()

        pos_elecs_entries = [line for line in cumulative_ttest_data_raw if line[-1] > 2.0]
        neg_elecs_entries = [line for line in cumulative_ttest_data_raw if line[-1] < -2.0]

        pos_elec_tags = self.extract_single_elec_tag(bipolar_ttest_data=pos_elecs_entries)
        neg_elec_tags = self.extract_single_elec_tag(bipolar_ttest_data=neg_elecs_entries)

        pos_monopol_indices = self.find_monopolar_indices_for_elecs(tal_struct, pos_elec_tags)

        neg_monopol_indices = self.find_monopolar_indices_for_elecs(tal_struct, neg_elec_tags)

        print pos_elec_tags
        print neg_elec_tags

        p = list(pos_monopol_indices)  # pos elec indices
        n = list(neg_monopol_indices)  # neg elec indices

        np_set = set(n + p)  # set of pos and neg indices

        tot_set = set(list(range(tal_struct.shape[0])))  # set of all indices    

        non_sig_elec_indices = list(tot_set - np_set)  # non-significen elecs indices

        w = BrainPlotOffscreenWidget()
        w.set_size(1000, 1000)

        lh = Hemisphere(hemi='l')
        rh = Hemisphere(hemi='r')

        w.add_display_object('lh', lh)
        w.add_display_object('rh', rh)

        # dg_elec = Electrodes(shape='sphere')

        pos_tal_struct = tal_struct[p]
        neg_tal_struct = tal_struct[n]
        non_sig_tal_struct = tal_struct[non_sig_elec_indices]

        sg_pos_tal_struct = pos_tal_struct[(pos_tal_struct.eType == 'S') | (pos_tal_struct.eType == 'G')]
        sg_neg_tal_struct = neg_tal_struct[(neg_tal_struct.eType == 'S') | (neg_tal_struct.eType == 'G')]

        sg_non_sig_tal_struct = non_sig_tal_struct[
            (non_sig_tal_struct.eType == 'S') | (non_sig_tal_struct.eType == 'G')]

        pos_elec = None
        neg_elec = None

        if sg_pos_tal_struct.shape[0]:
            sg_pos_elec_locs = np.vstack(
                (sg_pos_tal_struct.avgSurf.x, sg_pos_tal_struct.avgSurf.y, sg_pos_tal_struct.avgSurf.z)).T
            sg_pos_elec = Electrodes(shape='sphere')
            sg_pos_elec.set_electrodes_locations(loc_array=sg_pos_elec_locs)
            sg_pos_elec.set_electrodes_color(c=self.pos_color)
            w.add_display_object('sg_pos_elec', sg_pos_elec)

        if sg_neg_tal_struct.shape[0]:
            sg_neg_elec_locs = np.vstack(
                (neg_tal_struct.avgSurf.x, neg_tal_struct.avgSurf.y, neg_tal_struct.avgSurf.z)).T
            sg_neg_elec = Electrodes(shape='sphere')
            sg_neg_elec.set_electrodes_locations(loc_array=sg_neg_elec_locs)
            sg_neg_elec.set_electrodes_color(c=self.neg_color)
            w.add_display_object('sg_neg_elec', sg_neg_elec)

        if sg_non_sig_tal_struct.shape[0]:
            sg_non_sig_elec_locs = np.vstack(
                (sg_non_sig_tal_struct.avgSurf.x, sg_non_sig_tal_struct.avgSurf.y, sg_non_sig_tal_struct.avgSurf.z)).T
            sg_non_sig_elec = Electrodes(shape='sphere')
            sg_non_sig_elec.set_electrodes_locations(loc_array=sg_non_sig_elec_locs)
            sg_non_sig_elec.set_electrodes_color(c=self.non_sig_color)
            w.add_display_object('sg_non_sig_elec', sg_non_sig_elec)

        for camera_name in ['left', 'right', 'bottom']:
            w.load_camera_settings(filename=camera_name + '.camera.json')
            image_path = self.get_path_to_resource_in_workspace('reports', 'brain-' + camera_name + '.png')
            w.take_screenshot(filename=image_path)
            # w.take_screenshot(filename='brain-'+camera_name+'.png')

        # axial plot

        d_pos_tal_struct = pos_tal_struct[(pos_tal_struct.eType == 'D')]
        d_neg_tal_struct = neg_tal_struct[(neg_tal_struct.eType == 'D')]
        d_non_sig_tal_struct = non_sig_tal_struct[(non_sig_tal_struct.eType == 'D')]

        d_pos_elec_locs = np.vstack(
            (d_pos_tal_struct.avgSurf.x, d_pos_tal_struct.avgSurf.y, d_pos_tal_struct.avgSurf.z)).T
        d_neg_elec_locs = np.vstack(
            (d_neg_tal_struct.avgSurf.x, d_neg_tal_struct.avgSurf.y, d_neg_tal_struct.avgSurf.z)).T
        d_non_sig_elec_locs = np.vstack(
            (d_non_sig_tal_struct.avgSurf.x, d_non_sig_tal_struct.avgSurf.y, d_non_sig_tal_struct.avgSurf.z)).T

        w_axial = BrainPlotOffscreenWidget()
        w_axial.set_size(1000, 1000)

        # axial_slice = AxialSlice(fname='/Users/m/RAM_PLOTS_GIT/datasets/axial-mni-7.0.vtk')
        axial_slice = AxialSlice(fname=locate_file(filename='axial-tal-13.0.vtk',local_dir='datasets'))

        w_axial.add_display_object('axial_slice', axial_slice)

        d_elecs_pos = project_multiple_electrodes_onto_plane(elecs_pos=d_pos_elec_locs, axial_slice=axial_slice,
                                                             max_distance=30.0)
        d_elecs_neg = project_multiple_electrodes_onto_plane(elecs_pos=d_neg_elec_locs, axial_slice=axial_slice,
                                                             max_distance=30.0)
        d_elecs_non_sig = project_multiple_electrodes_onto_plane(elecs_pos=d_non_sig_elec_locs, axial_slice=axial_slice,
                                                                 max_distance=30.0)


        if d_elecs_pos.shape[0]:
            d_elecs_pos_obj = Electrodes(shape='sphere')
            d_elecs_pos_obj.set_electrodes_locations(loc_array=np.array(d_elecs_pos))
            d_elecs_pos_obj.set_electrodes_color(c=self.pos_color)
            w_axial.add_display_object('d_elecs_pos_obj', d_elecs_pos_obj)

        if d_elecs_neg.shape[0]:
            d_elecs_neg_obj = Electrodes(shape='sphere')
            d_elecs_neg_obj.set_electrodes_locations(loc_array=np.array(d_elecs_neg))
            d_elecs_neg_obj.set_electrodes_color(c=self.neg_color)
            w_axial.add_display_object('d_elecs_neg_obj', d_elecs_neg_obj)

        if d_elecs_non_sig.shape[0]:
            d_elecs_non_sig_obj = Electrodes(shape='sphere')
            d_elecs_non_sig_obj.set_electrodes_locations(loc_array=np.array(d_elecs_non_sig))
            d_elecs_non_sig_obj.set_electrodes_color(c=self.non_sig_color)
            w_axial.add_display_object('d_elecs_non_sig_obj', d_elecs_non_sig_obj)

        for camera_name in ['bottom']:
            w_axial.load_camera_settings(filename=camera_name + '.camera.json')
            image_path = self.get_path_to_resource_in_workspace('reports', 'axial-' + camera_name + '.png')
            w_axial.take_screenshot(filename=image_path)


if __name__ == '__main__':
    # algorithm to select indices of elements that are shared between two arrays
    import numpy as np

    x = np.array([3, 5, 7, 1, 9, 8, 6, 6])
    y = np.array([2, 1, 5, 10, 100, 6])

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
