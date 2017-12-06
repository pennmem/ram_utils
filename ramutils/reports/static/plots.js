/**
 * Plotting functions for RAM reports.
 */

var ramutils = (function (mod, Plotly) {
  /**
   * Make the axes options object.
   * @param {String} xlabel
   * @param {String} ylabel
   * @return {Object}
   */
  mod.makeAxesOptions = function (xlabel, ylabel) {
    return {
      xaxis: {
        title: xlabel
      },
      yaxis: {
        title: ylabel
      }
    };
  },

  mod.plots = {
    /**
     * Plots serial position curves.
     * @param {Array} serialPos - Serial positions (x-axis)
     * @param {Array} overallProbs - Overall probabilities per serial position
     * @param {Array} firstProbs - Probability of first recall per serial position
     */
    serialpos: function (serialPos, overallProbs, firstProbs) {
      const mode = "lines+markers";
      const data = [
        {
          x: serialPos,
          y: overallProbs,
          mode: mode,
          name: "Overall"
        },
        {
          x: serialPos,
          y: firstProbs,
          mode: mode,
          name: "First recall"
        }
      ];

      const layout = mod.makeAxesOptions('Serial position', 'Probability');
      layout.xaxis.range = [1, 12];
      layout.yaxis.range = [0, 1];

      Plotly.plot('serialpos-plot-placeholder', data, layout);
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
    }
  };

  return mod;
})(ramutils || {}, Plotly);
