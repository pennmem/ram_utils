from RamPipeline import *

import numpy as np
import pandas as pd

from sklearn.externals import joblib


class CountSessions(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.region_session_total = None
        self.area_session_total = None
        #self.region_session_significant = None
        #self.area_session_significant = None

    def restore(self):
        self.region_session_total = joblib.load(self.get_path_to_resource_in_workspace('region_session_total.pkl'))
        self.pass_object('region_session_total', self.region_session_total)

        self.area_session_total = joblib.load(self.get_path_to_resource_in_workspace('area_session_total.pkl'))
        self.pass_object('area_session_total', self.area_session_total)

        #self.region_session_significant = joblib.load(self.get_path_to_resource_in_workspace('region_session_significant.pkl'))
        #self.pass_object('region_session_significant', self.region_session_significant)

        #self.area_session_significant = joblib.load(self.get_path_to_resource_in_workspace('area_session_significant.pkl'))
        #self.pass_object('area_session_significant', self.area_session_significant)

    def run(self):
        ps_table = self.get_passed_object('ps_table')

        regions = sorted(ps_table['Region'].unique())
        areas = sorted(ps_table['Area'].unique())
        areas.remove('')

        print 'Regions:', regions
        print 'Areas:', areas

        self.region_session_total = dict.fromkeys(regions,0)
        self.area_session_total = dict.fromkeys(areas,0)

        stim_groups = ps_table.groupby(['Subject','stimAnodeTag','stimCathodeTag'])

        for idx,gr in stim_groups:
            region = gr['Region'].unique()
            if len(region) > 1:
                print 'ERROR: Multiple regions for %s %s-%s' % idx
            region = region[0]
            self.region_session_total[region] += 1

            area = gr['Area'].unique()
            if len(area) > 1:
                print 'ERROR: Multiple areas for %s %s-%s' % idx
            area = area[-1]
            if area != '':
                self.area_session_total[area] += 1

        #self.region_session_significant
        #self.area_session_significant

        #self.region_session_total = sorted(zip(self.region_session_total.keys(), self.region_session_total.values()))
        #self.area_session_total = sorted(zip(self.area_session_total.keys(), self.area_session_total.values()))

        self.pass_object('region_session_total', self.region_session_total)
        self.pass_object('area_session_total', self.area_session_total)

        joblib.dump(self.region_session_total, self.get_path_to_resource_in_workspace('region_session_total.pkl'))
        joblib.dump(self.area_session_total, self.get_path_to_resource_in_workspace('area_session_total.pkl'))
