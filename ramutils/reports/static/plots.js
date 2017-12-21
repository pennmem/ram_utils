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
    plotSerialpos: function (serialPos, overallProbs, firstProbs) {
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

    plotCategoricalFreeRecallSummary: function (irt_within_cat, irt_between_cat, repetition_ratios, subject_ratio) {
      const data = [
        {
          x: ['Within Cat', 'Between Cat'],
          y: [irt_within_cat, irt_between_cat],
          width: [0.5, 0.5],
          type: 'bar',
        },
        {
          x: repetition_ratios,
          type: 'histogram',
          xaxis: 'x2',
          yaxis: 'y2'
        },
        {
          x: [subject_ratio, subject_ratio],
          y: [0, 30],
          type: 'scatter',
          mode: 'lines',
          line: {
            color: 'black',
            dash: 'dot'
          },
          xaxis: 'x2',
          yaxis: 'y2'
        }
      ];

      const layout = {
        xaxis: {
          domain: [0, .45]
        },
        yaxis: {
          title: 'IRT [msec]'
        },
        xaxis2: {
          domain: [.55, 1]
        },
        yaxis2: {
          title: '# of Subjects',
          anchor: 'x2'
        },
        width: 1000,
        height: 500,
        showlegend: false
      };

      Plotly.plot('catfr-placeholder', data, layout);

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
    plotClassifierPerformance: function (fpr, tpr, low, middle, high, tags) {
      let data = [
        {
          x: [0, 1],
          y: [0, 1],
          mode: 'lines',
          line: {
            color: 'black',
            dash: 'dot'
          },
          showlegend: false
        },
      ];

      for (i = 0; i < fpr.length; i++) {
        const tag = tags[i];
        const roc_curve = {
          x: fpr[i],
          y: tpr[i],
          type: 'scatter',
          mode: 'lines',
          name: `${tag} ROC Curve`
        };
        const tercile = {
          x: ['low', 'middle', 'high'],
          y: [low[i], middle[i], high[i]],
          xaxis: 'x2',
          yaxis: 'y2',
          type: 'bar',
          name: `${tag} Tercile`
        };
        data.push(roc_curve);
        data.push(tercile);
      }

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

        showlegend: true,

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
    },

    /**
     * Plot classifier output as a function of amplitude for each stim site in
     * PS4. Each parameter (other than amplitude) should have keys that match
     * the stim channel labels.
     * @param {Array} amplitude
     * @param {Object} encoding
     * @param {Object} distract
     * @param {Object} retrieval
     * @param {Object} sham
     * @param {Object} postStim
     */
    plotPS4ClassifierOutput: function(amplitude, encoding, distract, retrieval, sham, postStim) {
      const labels = Object.keys(encoding);

      const deltaClassifierData = (() => {
        let data = [];

        for (let i = 0; i < 2; i++) {
          const xaxis = i == 0 ? 'x' : 'x2';
          data.push(
            {
              x: amplitude,
              y: encoding[labels[i]],
              mode: 'markers',
              name: 'Encoding',
              xaxis: xaxis
            },
            {
              x: amplitude,
              y: distract[labels[i]],
              mode: 'markers',
              name: 'Distract',
              xaxis: xaxis
            },
            {
              x: amplitude,
              y: retrieval[labels[i]],
              mode: 'markers',
              name: 'Retrieval',
              xaxis: xaxis
            },
            {
              x: amplitude,
              y: sham[labels[i]],
              mode: 'markers',
              name: 'Sham',
              xaxis: xaxis
            }
          );
        }

        return data;
      })();

      const postData = (() => {
        return [
          {
            x: amplitude,
            y: postStim[labels[0]],
            mode: 'markers',
            name: labels[0]
          },
          {
            x: amplitude,
            y: postStim[labels[1]],
            mode: 'markers',
            xaxis: 'x2',
            // yaxis: 'y2',
            name: labels[1]
          }
        ];
      })();

      const layout = {
        xaxis: {
          title: "Amplitude [mA]",
          domain: [0, 0.45]
        },
        xaxis2: {
          title: "Amplitude [mA]",
          domain: [0.55, 1]
        },
        yaxis: {
          title: "Delta classifier (post minus pre)"
        }
      };

      Plotly.plot('ps4-delta-classifier-placeholder', deltaClassifierData, layout);
      Plotly.plot('ps4-post-classifier-placeholder', postData, layout);
    }
  };

  return mod;
})(ramutils || {}, Plotly);
