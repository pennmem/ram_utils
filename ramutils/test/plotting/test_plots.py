import functools
import numpy as np
import matplotlib

# No graphical display. Must be called before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ramutils.plotting import plots
from pkg_resources import resource_filename

datafile = functools.partial(resource_filename,
                             'ramutils.test.test_data')

def test_recall_prob_plot():
    spos = np.arange(1,13,1)
    recall_prob_stim = np.random.rand(12)
    recall_prob_nostim = np.random.rand(12)
    recall_prob = [recall_prob_stim, recall_prob_nostim]

    try:
        ax = plots.probability_lineplot(spos, recall_prob)
    except RuntimeError:
        pass # to be expected since no labels were passed

    ax = plots.probability_lineplot(spos,
                                    recall_prob,
                                    labels=['Stim Items', 'Non-stim Items'])
    plt.savefig(datafile("prob_of_recall.png"))
    return


def test_roc_plot():
    fpr = np.random.rand(50)
    tpr = np.random.rand(50)
    ax = plots.roc_curve(fpr, tpr)
    plt.savefig(datafile("roc_curve.png"))
    return


def test_repetition_ratio():
    irt_within_cat = 20000
    irt_between_cat = 7500
    plots.repeitition_ratio(irt_within_cat, irt_between_cat)
    plt.savefig(datafile("irt_barplot.png"))
    return


def test_tercile_classifier():
    plots.tercile_classifier_estimate([-30, 10, 20])
    plt.savefig(datafile("tercile_classifier_barplot.png"))
    return

def test_delta_recall():
    percent_diff_from_mean = (20, -5)
    plots.delta_recall(percent_diff_from_mean)
    plt.savefig(datafile("delta_recall_barplot.png"))
    return

def test_stim_and_recall():
    n_stims = np.random.randint(12, size=25)
    n_stim_recall = np.random.randint(12, size=13)
    n_nostim_recall = np.random.randint(12, size=12)
    stim_lists = np.arange(1, 26, 2)
    nostim_lists = np.arange(2, 26, 2)
    plots.stim_and_recall(25, n_stims, stim_lists, n_stim_recall, nostim_lists, n_nostim_recall)
    plt.savefig(datafile("stim_and_recall.png"))
    return

def test_biomarker_distributions():
    probs = np.random.normal(loc=.5, scale=.1, size=2000)
    plots.biomarker_distribution(probs, "Pre-stim classifier output")
    plt.savefig(datafile("biomarker_distribution.png"))
    return

def test_eeg_data():
    eeg_data = np.random.random(size=(128, 800, 3))
    plots.post_stim_eeg(eeg_data)
    plt.savefig(datafile("post_stim_eeg.png"))
    return