/**
 * Plotting functions for RAM reports.
 */

var ramutils = (function (mod, Plotly) {
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
      let mode = "lines+markers";
      let data = [
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
      let layout = mod.makeAxesOptions('Serial position', 'Probability');
      layout.xaxis.range = [1, 12];
      layout.yaxis.range = [0, 1];

      Plotly.plot('serialpos-placeholder', data, layout);
    }
  };

  return mod;
})(ramutils || {}, Plotly);
