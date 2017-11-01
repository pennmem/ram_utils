\section*{<SUBJECT> RAM <TASK> Free Recall Report}

\begin{tabular}{ccc}
\begin{minipage}[htbp]{170pt}
In the free recall task, participants are presented with a list of words, one after the other, and later asked to recall as many words as possible in any order.
\begin{itemize}
    \item\textbf{Number of sessions: }$<NUMBER_OF_SESSIONS>$
    \item\textbf{Number of electrodes: }$<NUMBER_OF_ELECTRODES>$
    \item\textbf{FR1 Area Under Curve: }$<AUC>$
    \item\textbf{FR1 Permutation test $p$-value:} $<PERM-P-VALUE>$
    \end{itemize}
\end{minipage}
&
\begin{tabular}{|c|c|c|c|c|c|}
\hline Session & Date & Length (min) & \#lists & Perf & Amp (mA) \\
<SESSION_DATA>
\end{tabular}
\end{tabular}

\vspace{1pc}

\begin{center}
\textbf{\Large <ROC_TITLE>}
\end{center}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.25]{<ROC_AND_TERC_PLOT_FILE_1>}
\caption{\textbf{(a)} ROC curve for the subject;
\textbf{(b)} Subject recall performance represented as
percentage devation from the (subject) mean, separated by tercile
of the classifier encoding efficiency estimate for each encoded word.}
\end{figure}

$\bullet$ Area Under Curve = $<AUC-1>$

$\bullet$ Permutation test $p$-value = $<PERM-P-VALUE-1>$

$\bullet$ Median of classifier output = $<JSTAT-THRESH-1>$

\begin{figure}[!h]
\centering
\includegraphics[scale=0.25]{<ROC_AND_TERC_PLOT_FILE_2>}
\caption{\textbf{(a)} ROC curve for the subject;
\textbf{(b)} Subject recall performance represented as
percentage devation from the (subject) mean, separated by tercile
of the classifier encoding efficiency estimate for each encoded word.}
\end{figure}

$\bullet$ Area Under Curve = $<AUC-2>$

$\bullet$ Permutation test $p$-value = $<PERM-P-VALUE-2>$

$\bullet$ Median of classifier output = $<JSTAT-THRESH-2>$

%\begin{center}
%\textbf{\Large <STIM_TITLE>}
%\end{center}

%\begin{figure}[!h]
%    \begin{subfigure}[b]{.30\textwidth}
%        \includegraphics[width=\linewidth, height=.2\textheight]{<ESTIMATED_STIM_EFFECT_PLOT_FILE_list>}
%    \end{subfigure}\hfill
%    \begin{subfigure}[b]{.30\textwidth}
%        \includegraphics[width=\linewidth, height=.2\textheight]{<ESTIMATED_STIM_EFFECT_PLOT_FILE_stim>}
%    \end{subfigure}\hfill
%    \begin{subfigure}[b]{.30\textwidth}
%        \includegraphics[width=\linewidth, height=.2\textheight]{<ESTIMATED_STIM_EFFECT_PLOT_FILE_post_stim>}
%    \end{subfigure}\hfill
%    \caption*{\textbf{Estimated effect of stim:} Filled circles represent point estimates
%    for the effect of stimulation, while horizontal lines are the 95\% credible intervals.
%    To get odds ratios from the point estimates, calcualte $e^{\beta}$ where $\beta$ is 
%    the coefficient of interest. Estimates come from a hierarchical model predicting
%    word-level recall from stimulation controlling for serial position effects. 
%    Session-specific effects are assumed to come from a common group distribution
%    centered around the cross-session average. The model is fit using MCMC sampling
%    with 5,000 draws.}
%\end{figure}

<REPORT_PAGES>

\clearpage
\section*{Data Quality Metrics}
\subsection*{Biomarker Distributions}
\begin{figure}[!h]
\centering
\includegraphics[scale=0.5]{<BIOMARKER_HISTOGRAM>}
\end{figure}

%\begin{figure}[!h]
%\centering
%\includegraphics[scale=0.5]{<DELTA_CLASSIFIER_HISTOGRAM>}
%\end{figure}
\subsection*{EEG data}
\vspace{-1cm}
\begin{figure}[!h]
\centering
\includegraphics[scale=0.5]{<POST_STIM_EEG>}
\caption*{Voltage during the post-stimulation period, averaged across trials.\\
          Voltages beyond +-500 $\mu$V not shown; voltages between -20 and 20 $mu$V are in white.}
\end{figure}

\end{document}
