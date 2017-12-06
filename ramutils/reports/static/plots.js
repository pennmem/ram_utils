/**
 * Plotting functions for RAM reports.
 */

var ramutils = (function (mod, Plotly) {
  mod.plots = {
    /**
     * Plots serial position curves.
     * @param {Array} serialPos - Serial positions (x-axis)
     * @param {Object} overallProbs - Overall probabilities per serial position
     * @param {Object} firstProbs - Probability of first recall per serial position
     */
    serialpos: function (serialPos, overallProbs, firstProbs) {
      const mode = "lines+markers";
      let data = [];

      for (let name in overallProbs) {
        data.push({
          x: serialPos,
          y: overallProbs[name],
          mode: mode,
          name: name
        });
      }

      for (let name in firstProbs) {
        data.push({
          x: serialPos,
          y: firstProbs[name],
          mode: mode,
          name: name
        });
      }

      const layout = {
        xaxis: {
          title: 'Serial position',
          range: [0.9, 12.1]
        },
        yaxis: {
          title: 'Probability',
          range: [0, 1]
        }
      };

      Plotly.plot('serialpos-plot-placeholder', data, layout);
    },

    /**
     * Plot a summary of recall of stimed/non-stimed items.
     * @param {Object} nonStimRecalls
     * @param {Object} stimRecalls
     * @param {Object} stimEvents
     */
    plotRecallSummary: function (nonStimRecalls, stimRecalls, stimEvents) {
      const data = [
        {
          x: nonStimRecalls.listno,
          y: nonStimRecalls.recalled,
          mode: 'markers',
          type: 'scatter',
          marker: {size: 12},
          name: 'Non-stim recalls'
        },
        {
          x: stimRecalls.listno,
          y: stimRecalls.recalled,
          mode: 'markers',
          type: 'scatter',
          marker: {size: 12},
          name: 'Stim recalls'
        },
        {
          x: stimEvents.listno,
          y: stimEvents.count,
          type: 'bar',
          name: 'Stim events'
        }
      ];

      const layout = {
        xaxis: {
          title: 'List number',
          range: [1, Math.max(...nonStimRecalls.listno.concat(stimRecalls.listno)) + 0.5]
        },
        yaxis: {
          title: 'Number of items',
          range: [0, 12.1]
        }
      };

      Plotly.plot('stim-recall-placeholder', data, layout);
    },

    /**
     * Plot probability of stimulation versus serial position.
     * @param {Array} serialpos - serial position
     * @param {Array} prob - probabilities
     */
    plotStimProbability: function (serialpos, prob) {
      const data = [{
        x: serialpos,
        y: prob,
        mode: 'lines+markers',
        name: 'Probability'
      }];

      const layout = {
        xaxis: {
          title: 'Serial position',
          range: [0.5, 12.5]
        },
        yaxis: {
          title: 'Probability of stimulation',
          range: [0, 1]
        }
      };

      Plotly.plot('stim-probability-placeholder', data, layout);
    },

    /**
     * Plot overall recall difference for stim/post-stim items.
     */
    plotRecallDifference: function (stimPercent, postStimPercent) {
      const data = [{
        x: ['stim', 'post-stim'],
        y: [stimPercent, postStimPercent],
        type: 'bar'
      }];

      const layout = {
        xaxis: {title: 'Items'},
        yaxis: {
          title: 'Recall difference [%]',
          range: [-65, 65]
        }
      };

      Plotly.plot('recall-difference-placeholder', data, layout);
    },

    /**
     * Classifier performance plots: ROC curve and tercile plot.
     * @param {Array} fpr - False positive rate
     * @param {Array} tpr - True positive rate
     * @param {Number} low
     * @param {Number} middle
     * @param {Number} high
     */
    classifierPerformance: function (fpr, tpr, low, middle, high) {
      const data = [
        {
          x: fpr,
          y: tpr,
          type: 'scatter',
          mode: 'lines',
          name: 'ROC'
        },
        {
          x: ['low', 'middle', 'high'],
          y: [low, middle, high],
          xaxis: 'x2',
          yaxis: 'y2',
          type: 'bar',
          name: 'Tercile'
        }
      ];

      const layout = {
        xaxis: {
          title: 'False positive rate',
          domain: [0, 0.45],
          range: [0, 1]
        },
        yaxis: {
          title: 'True positive rate',
          range: [0, 1]
        },
        xaxis2: {
          title: 'Tercile of classifier estimate',
          domain: [0.55, 1]
        },
        yaxis2: {
          title: 'Recall change from mean (%)',
          anchor: 'x2'
        },

        showlegend: false,

        // FIXME: responsive and square aspect ratio?
        width: 1000,
        height: 500
      };

      Plotly.plot('classifier-performance-plot-placeholder', data, layout);
    },

    /**
     * Plot classifier output distributions.
     * @param {Array} preStim - classifier output pre-stim
     * @param {Array} postStim - classifier output post-stim
     */
    plotClassifierOutputDistros: function (preStim, postStim) {
      let delta = [];
      for (let i in preStim) {
        delta.push(postStim[i] - preStim[i]);
      }

      const data = [
        {
          x: preStim,
          type: 'histogram',
          name: 'Pre-stim'
        },
        {
          x: postStim,
          type: 'histogram',
          name: 'Post-stim'
        },
        {
          x: delta,
          type: 'histogram',
          name: 'Post minus pre'
        }
      ];

      const layout = {
        // barmode: 'overlay',
        xaxis: {title: 'Classifier output'},
        yaxis: {title: 'Frequency'}
      };

      Plotly.plot('classifier-output-placeholder', data, layout);
    }
  };

  return mod;
})(ramutils || {}, Plotly);
