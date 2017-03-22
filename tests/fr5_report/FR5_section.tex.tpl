{% block FR5_SECTION %}
\section*{{{SUBJECT}} RAM FR5 Free Recall Report}

\begin{tabular}{ccc}
\begin{minipage}[htbp]{170pt}
In the free recall task, participants are presented with a list of words, one after the other, and later asked to recall as many words as possible in any order.
\begin{itemize}
    \item\textbf{Number of sessions: }${{NUMBER_OF_SESSIONS}}$
    \item\textbf{Number of electrodes: }${{NUMBER_OF_ELECTRODES}}$
    \item\textbf{FR1 Area Under Curve: }${{AUC}}$\%
    \item\textbf{FR1 Permutation test $p$-value:} ${{PERM-P-VALUE}}$
    \end{itemize}
\end{minipage}
&
\begin{tabular}{|c|c|c|c|c|c|}
\hline Session & Date & Length (min) & \#lists & Perf & Amp (mA) \\
{{FR5_SESSION_DATA}}
\end{tabular}
\end{tabular}

\vspace{3pc}

\begin{center}
\textbf{\Large Classifier generalization to FR3}
\end{center}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.45]{{{ROC_AND_TERC_PLOT_FILE}}}
\caption{\textbf{(a)} ROC curve for the subject;
\textbf{(b)} Subject recall performance represented as
percentage devation from the (subject) mean, separated by tercile
of the classifier encoding efficiency estimate for each encoded word.}
\end{figure}

$\bullet$ Area Under Curve = ${{FR3-AUC}}$\%

$\bullet$ Permutation test $p$-value ${{FR3-PERM-P-VALUE}}$


{% for session in FR5_SESSION_SUMMARIES %}
{% if session != -1 %}

\clearpage

\begin{center}
\textbf{\Large {{FR5_SESSION_SUMMARIES[session].STIMTAG}} ({{FR5_SESSION_SUMMARIES[session].REGION}} ), 200 Hz, Session: {{session}} }
\end{center}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c|c}
\multicolumn{4}{c}{\textbf{Free Recall}} \\
\hline
${{N_WORDS[session]}}$ words & ${{N_CORRECT_WORDS[session]}}$ correct (${{PC_CORRECT_WORDS[session}}$\%) &${{N_PLI[session]$ PLI (${{PC_PLI[session]}}$\%) &${{N_ELI[session]}}$ ELI (${{PC_ELI[session]}}$\%) \\ \hline
\end{tabular}
\caption{An intrusion was a word that was vocalized during the retrieval period that was not studied on the most recent list. Intrusions were either words from a previous list (\textbf{PLI}: prior-list intrusions) or words that were not studied at all (\textbf{ELI}: extra-list intrusions).}
\end{table}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c}
\multicolumn{3}{c}{\textbf{Math distractor}} \\
\hline
${{N_MATH[session]}}$ math problems & ${{N_CORRECT_MATH[session]}}$ correct (${{PC_CORRECT_MATH[session]}}$\%) & ${{MATH_PER_LIST[session]}}$ problems per list  \\ \hline
\end{tabular}
\caption{After each list, the patient was given 20 seconds to perform as many arithmetic problems as possible, which served as a distractor before the beginning of recall.}
\end{table}

\begin{figure}[!h]
\centering
\textbf{Probability of Recall}

\subfigure{\includegraphics[scale=0.4]{{{PROB_RECALL_PLOT_FILE[session]}}}}
\caption{\textbf{Free recall:}
(a) Overall probability of recall as a function of serial position.
(b) Probability of first recall as a function of serial position.}
\end{figure}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c|c}
\multicolumn{4}{c}{\textbf{Stim vs Non-Stim Recalls}} \\
\hline
${{N_CORRECT_STIM[session]}}/{{N_TOTAL_STIM[session]}}$ (${{PC_FROM_STIM[session]}}$\%) from stim lists & ${{N_CORRECT_NONSTIM[session]}}/{{N_TOTAL_NONSTIM[session]}}$ (${{PC_FROM_NONSTIM[session]}}$\%) from non-stim lists & $\chi^2(1)={{CHISQR[session]}}$ & $p={{PVALUE[session]}}$ \\
\hline
 {{ITEMLEVEL_COMPARISON[session]}}
\end{tabular}
\end{table}

\begin{table}[!h]
\centering
\begin{tabular}{c|c}
\multicolumn{2}{c}{\textbf{Stim vs Non-Stim Intrusions}} \\
\hline
${{N_STIM_INTR[session]}}/{{N_TOTAL_STIM[session]}}$ (${{PC_FROM_STIM_INTR[session]}}$\%) from stim lists & ${{N_NONSTIM_INTR[session]}}/{{N_TOTAL_NONSTIM[session]}}$ (${{PC_FROM_NONSTIM_INTR[session]}}$\%) from non-stim lists \\
\hline
\end{tabular}
\end{table}
\clearpage

\begin{center}
\textbf{\Large Stim and recall analysis}
\end{center}

\begin{figure}[!h]
\centering
\subfigure{\includegraphics[height=0.25\textheight]{{{STIM_AND_RECALL_PLOT_FILE[session]}}}}
\caption*{\textbf{Stim and recall:} Number of stims and recalled items per list. Red circles represent the number of recalled items from stim lists. Blue circles represent the number of recalled items from non-stim lists. Bars represent the number of stims per list.}
\textbf{Probability of Stimulation}

\subfigure{\includegraphics[scale=0.3]{{{PROB_STIM_PLOT_FILE[session]}} !}}
\caption*{Probability of stimulation as a function of serial position.}
\end{figure}

{% if STIM_VS_NON_STIM_HALVES_PLOT_FILE[session] %}
% \vspace{1pc}

\begin{figure}[!ht]
\centering
\includegraphics[height=0.25\textheight]{{{STIM_VS_NON_STIM_HALVES_PLOT_FILE[session]}}}
\caption*{\textbf{(a)} Change in classifier output from the stimulated item (0-1366 ms) to the post-stimulated item (0-1366 ms) plotted for all items and for low-classifier output items. The control group comprises all items from non-stim lists.
\textbf{(b)} Change in recall performance for stimulated items contingent upon whether the stimulated item was presented during a low classifier state or a high classifier state (0-1366 ms).
\textbf{(c)} Change in recall performance for post-stimulated items contingent upon whether the stimulated item was presented during a low classifier state or a high classifier state (0-1366 ms).}
\end{figure}
{% end %}

{% end %}
{% endblock %}