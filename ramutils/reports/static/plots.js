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
        console.log(name);
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
        title: "Probability of Recall as a Function of Serial Position",
        xaxis: {
          title: 'Serial position',
          range: [0.9, serialPos.length + .1]
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
        title: 'Number of Items Stimulated and Number of Items Recalled',
        xaxis: {
          title: 'List number',
        },
        yaxis: {
          title: 'Number of items',
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
        title: "Probability of Stimulation as a Function of Serial Position",
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
        title: "Change in Recall Performance for Stimulated and Post-Stimulation Items",
        xaxis: {title: 'Items'},
        yaxis: {
          title: 'Recall difference [%]',
          range: [Math.min(stimPercent, postStimPercent, -4) - 1, Math.max(stimPercent, postStimPercent, 0) + 1]
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
          name: `${tag} ROC Curve`,
          legendgroup: i
        };
        const tercile = {
          x: ['low', 'middle', 'high'],
          y: [low[i], middle[i], high[i]],
          xaxis: 'x2',
          yaxis: 'y2',
          type: 'bar',
          name: `${tag} Tercile`,
          legendgroup: i
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
        legend: {
            x: 1,
        },

        showlegend: true,

        // FIXME: responsive and square aspect ratio?
        width: 1000,
        height: 500
      };

      Plotly.plot('classifier-performance-plot-placeholder', data, layout);
    },
    /**
    * Plot classifier weights as a heatmap: frequency by channel
    * This function assumes that the classifier weights are log-spaced.
    * @param{Array} weights - classfier weights
    * @param{Array} freqs - frequencies used
    */
    plotClassifierWeights: function(weights, freqs, labels, names){

        let data =[];
        let layout = {
            title: 'Classifier Activation',
            showlegend: true
        }
        let zmin = Math.min(...[].concat(...[].concat(...[].concat(...weights))))
        let zmax = Math.max(...[].concat(...[].concat(...[].concat(...weights))))


        for(i=0;i<weights.length;i++){
            let these_freqs = freqs[i];
            let xaxis = 'x';
            let yaxis = 'y';
            if (i>0) {
             xaxis = xaxis+(i+1);
             yaxis = yaxis+(i+1);
             }

            data.push( {
                z: weights[i],
                y: these_freqs,
                type: 'heatmap',
                xaxis: xaxis,
                yaxis: yaxis,
                zmin: zmin,
                zmax: zmax,
                showscale: ((i+1)==weights.length),
                colorbar:{
                    title: 'Weight'
                    }
                });
            let xax_name = i==0? 'xaxis' : 'xaxis'+(i+1);
            let yax_name = i==0? 'yaxis' : 'yaxis'+(i+1);
            let start_offset = i==0?0:0.1/(weights.length);
            let end_offset = i==weights.length-1?0:0.1/weights.length;
            let freq_names = [];
            for(j=0; j<these_freqs.length; j++){
                freq_names.push(Math.round(these_freqs[j]).toString())
                }

            layout[xax_name] = {title: names[i],
                                domain:[i/weights.length+start_offset, (i+1)/weights.length-end_offset],
                                ticktext:labels[i],
                                tickvals:Array.from(Array(labels[i].length).keys()),
                                tickangle: -45,
                                };
            layout[yax_name] = {title: i?'':'Frequency (Hz)',
                                type:'log',
                                autorange:true,
                                anchor:xaxis,
                                tickmode:'linear',
                                tick0:Math.log10(these_freqs[0]),
                                dtick:Math.log10(
                                    these_freqs[these_freqs.length-1]/these_freqs[0]
                                    )/(these_freqs.length-1),
                                tickformat: '.0f'
                                };
        }
        Plotly.plot("classifier-weight-plot",data,layout)
    },

    /**
     * Plot classifier output distributions.
     * @param {Array} preStim - classifier output pre-stim
     * @param {Array} postStim - classifier output post-stim
     * @param {String} plotName - hyphenated suffix of the <div> element
       the plot will live in
     */
    plotClassifierOutputDistros: function (preStim, postStim, plotName) {
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

      let layout = {
        // barmode: 'overlay',
        yaxis: {title: 'Frequency'}
      };

      let div_name='classifier-output-placeholder';

      if(plotName){
        div_name = div_name + '-' + plotName.toString();
        layout.title = plotName;
      }

      Plotly.plot(div_name, data, layout);
    },
    /** Plot the feature matrix
    * @params {Array} features
    */
    plotZtransPowers: function (features) {
        const data = [{
            z: features,
            type: 'heatmap',
            name: 'Features',
        }];

        Plotly.plot('feature-plot-placeholder',data);
    },

    /**
     * Plot classifier output as a function of amplitude for each stim site in
     * PS4. Each parameter (other than amplitude) should have keys that match
     * the stim channel labels.
     * @param {Object} encoding
     * @param {Object} distract
     * @param {Object} retrieval
     * @param {Object} sham
     * @param {Object} postStim
     */
    plotPS4ClassifierOutput: function(data_dict) {
      const labels = Object.keys(data_dict);

      const deltaClassifierData = (() => {
        let data = [];

        for (let i = 0; i < labels.length; i++) {
          const xaxis = i == 0 ? 'x' : 'x2';
          data.push(
            {
              x: data_dict[labels[i]]['amplitude']['ENCODING'],
              y: data_dict[labels[i]]['delta_classifier']['ENCODING'],
              mode: 'markers',
              name: 'Encoding',
              xaxis: xaxis,
            },
            {
              x: data_dict[labels[i]]['amplitude']['DISTRACT'],
              y: data_dict[labels[i]]['delta_classifier']['DISTRACT'],
              mode: 'markers',
              name: 'Distract',
              xaxis: xaxis,
            },
            {
              x: data_dict[labels[i]]['amplitude']['RETRIEVAL'],
              y: data_dict[labels[i]]['delta_classifier']['RETRIEVAL'],
              mode: 'markers',
              name: 'Retrieval',
              xaxis: xaxis,
            }
            // TODO: Add back once we actually produce sham data results
//            {
//              x: data_dict[labels[i]]['sham']['amplitude'],
//              y: data_dict[labels[i]]['sham']['delta_classifier'],
//              mode: 'markers',
//              name: 'Sham',
//              xaxis: xaxis
//            }
          );
        }

        return data;
      })();

      const postStimClassifierData = (() => {
        let post_data = [];
        for (let i = 0; i < labels.length; i++) {
          const xaxis = i == 0 ? 'x' : 'x2';
          post_data.push(
            {
              x: data_dict[labels[i]]['post_stim_amplitude']['ENCODING'],
              y: data_dict[labels[i]]['post_stim_biomarker']['ENCODING'],
              mode: 'markers',
              name: 'Encoding',
              xaxis: xaxis
            },
            {
              x: data_dict[labels[i]]['post_stim_amplitude']['DISTRACT'],
              y: data_dict[labels[i]]['post_stim_biomarker']['DISTRACT'],
              mode: 'markers',
              name: 'Distract',
              xaxis: xaxis
            },
            {
              x: data_dict[labels[i]]['post_stim_amplitude']['RETRIEVAL'],
              y: data_dict[labels[i]]['post_stim_biomarker']['RETRIEVAL'],
              mode: 'markers',
              name: 'Retrieval',
              xaxis: xaxis
            }
            // TODO: Add back once we actually produce sham data results
//            {
//              x: data_dict[labels[i]]['sham']['amplitude'],
//              y: data_dict[labels[i]]['sham']['delta_classifier'],
//              mode: 'markers',
//              name: 'Sham',
//              xaxis: xaxis
//            }
          );
        }
        return post_data
      })();

      const layout_top = {
        title: "Classifier Response as a Function of Amplitude",
        xaxis: {
          title: `Amplitude [mA] (${labels[0]})`,
          domain: [0, 0.45]
        },
        xaxis2: {
          title: `Amplitude [mA] (${labels[1]})`,
          domain: [0.55, 1]
        },
        yaxis: {
          title: "Delta classifier (post minus pre)"
        }
      };

      const layout_bottom = {
        xaxis: {
          title: `Amplitude [mA] (${labels[0]})`,
          domain: [0, 0.45]
        },
        xaxis2: {
          title: `Amplitude [mA] (${labels[1]})`,
          domain: [0.55, 1]
        },
        yaxis: {
          title: "Classifier Output"
        }
      };

      Plotly.plot('ps4-delta-classifier-placeholder', deltaClassifierData, layout_top);
      Plotly.plot('ps4-post-classifier-placeholder', postStimClassifierData, layout_bottom);
    },
    plotStimTStatHistogram: function(stim_tstat_data){
        let good_trace = {
            x: stim_tstat_data.good_tstats,
            type: "histogram",
            name: "Accepted channels"
        };
        let bad_trace = {
            x: stim_tstat_data.bad_tstats,
            type: "histogram",
            name: "Rejected channels"
        };
        let data=[good_trace, bad_trace];
        let layout = {
             barmode: "stack",
             xaxis: {name: "T-statistic"},
             yaxis: {name: "Channel count"}
             };
        Plotly.plot("stim-tstat-histogram", data, layout)
    },
    barPlot: function(data, labels) {
        let plot_data = [{
            x: labels,
            y: data,
            type: 'bar'
        }];

        Plotly.newPlot("barplot-placeholder", plot_data)
    }

  };

  return mod;
})(ramutils || {}, Plotly);
