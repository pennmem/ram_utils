import numpy as np
import matplotlib

# No graphical display. Must be called before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plotting import plots


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
    plt.savefig("./sample_plots/prob_of_recall.png")
    return


def test_roc_plot():
    fpr = np.random.rand(50)
    tpr = np.random.rand(50)
    ax = plots.roc_curve(fpr, tpr)
    plt.savefig("./sample_plots/roc_curve.png")
    return


def test_repetition_ratio():
    irt_within_cat = 20000
    irt_between_cat = 7500
    plots.repeitition_ratio(irt_within_cat, irt_between_cat)
    plt.savefig("./sample_plots/irt_barplot.png")
    return


def test_tercile_classifier():
    plots.tercile_classifier_estimate([-30, 10, 20])
    plt.savefig("./sample_plots/tercile_classifier_barplot.png")
    return