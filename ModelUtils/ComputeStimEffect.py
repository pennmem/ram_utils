import matplotlib as mpl
mpl.use('Agg') # allows matplotlib to work without x-windows (for RHINO)

import sys, traceback
import pymc3 as pm
import matplotlib.pyplot as plt
from RamPipeline import *
from sklearn.externals import joblib
from ModelUtils.HierarchicalModel import HierarchicalModel, HierarchicalModelPlots


class ComputeStimEffect(RamTask):
    """ Fit a multilvel model for estimating the effect of stimulation on recall """
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.model_types = ['list', 'stim', 'post_stim']
        self.trace = None
        self.stim_effect_estimate = None
    
    def restore(self):
        subject = self.pipeline.subject
        try:
            for model_type in self.model_types:
                trace = joblib.load(self.get_path_to_resource_in_workspace('-'.join([subject, model_type, 'trace.pkl'])))
                self.pass_object('trace_'+ model_type, trace)
        except (IOError, AssertionError):
            self.run()
        
        return

    def run(self):
        """ 
            Typically, plotting is done in GenerateReportTasks. 
            However, this causes issues with the custom plots for the models, so the plotting is done
            at the same time as the model fitting to get around all of the global state that is set
            in the custom matplotlib wrapper
        """
        subject = self.pipeline.subject
        fr_stim_table = self.get_passed_object('fr_stim_table')

        # Need to fit 3 models here. Stim list vs. non-stim list, stim items vs.
        # low bio non-stim items, and post-stim items vs. low bio non-stim items
        for model_type in self.model_types:
            try:
                model = HierarchicalModel(fr_stim_table, subject, self.pipeline.experiment, item_comparison=model_type)
                trace = model.fit(draws=5000, tune=1000)
                self.pass_object('trace_'+ model_type, self.trace)
                joblib.dump(trace, self.get_path_to_resource_in_workspace('-'.join([subject, model_type, 'trace.pkl'])))
            except Exception as e:
                print('Unable to fit the stim effect model for level {}'.format(model_type), e)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
                continue
            
            model_title_map = {
                'list': "Stim List vs. Non-stim List",
                'stim': 'Stim Items vs. Low Biomarker Non-stim Items',
                'post_stim': 'Post-stim Items vs. Low Biomarker Non-stim Items'
            }

            Plotter = HierarchicalModelPlots(trace)
            ax = Plotter.forestplot(model_title_map[model_type])
            forestplot_path = self.get_path_to_resource_in_workspace('reports/' + '_'.join([self.pipeline.subject, model_type, 'forestplot.pdf']))
            plt.savefig(forestplot_path,
                        format="pdf", 
                        dpi=300, 
                        bbox_inches='tight', 
                        pad_inches=.1)
            plt.close()
            self.pass_object('ESTIMATED_STIM_EFFECT_PLOT_FILE_' + model_type, forestplot_path)

            ax = Plotter.traceplot()
            plt.savefig(self.get_path_to_resource_in_workspace('reports/' + '_'.join([self.pipeline.subject, model_type, 'traceplot.pdf'])),
                        format="pdf",
                        dpi=300, 
                        bbox_inches='tight', 
                        pad_inches=.1)
            plt.close()
        return