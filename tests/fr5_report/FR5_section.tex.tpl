{% block FR5_SECTION %}
\section*{{{SUBJECT}} RAM FR5 Free Recall Report}

\begin{tabular}{ccc}
\begin{minipage}[htbp]{170pt}
In the free recall task, participants are presented with a list of words, one after the other, and later asked to recall as many words as possible in any order.
\begin{itemize}
    \item\textbf{Number of sessions: }${{NUMBER_OF_FR5_SESSIONS}}$
    \item\textbf{Number of electrodes: }${{NUMBER_OF_ELECTRODES}}$
    \item\textbf{FR1 Area Under Curve: }${{AUC}}$\%
    \item\textbf{FR1 Permutation test $p$-value:} ${{PERM_P_VALUE}}$
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
\textbf{\Large Classifier generalization to FR5}
\end{center}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.45]{{{ROC_AND_TERC_PLOT_FILE}}}
\caption{\textbf{(a)} ROC curve for the subject;
\textbf{(b)} Subject recall performance represented as
percentage devation from the (subject) mean, separated by tercile
of the classifier encoding efficiency estimate for each encoded word.}
\end{figure}

$\bullet$ Area Under Curve = ${{FR5_AUC}}$\%

$\bullet$ Permutation test $p$-value ${{FR5_PERM_P_VALUE}}$


{% for session in FR5_SESSION_SUMMARIES %}
{% if session != -1 %}
{% set SESSION_SUMMARY = FR5_SESSION_SUMMARIES[session] %}

\clearpage

\begin{center}
\textbf{\Large {{SESSION_SUMMARY.STIMTAG}} ({{SESSION_SUMMARIES.REGION}} ), 200 Hz, Session: {{session}} }
\end{center}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c|c}
\multicolumn{4}{c}{\textbf{Free Recall}} \\
\hline
${{SESSION_SUMMARY.N_WORDS}}$ words & ${{SESSION_SUMMARY.N_CORRECT_WORDS}}$ correct (${{SESSION_SUMMARY.PC_CORRECT_WORDS[session}}$\%) &${{SESSION_SUMMARY.N_PLI$ PLI (${{SESSION_SUMMARY.PC_PLI}}$\%) &${{SESSION_SUMMARY.N_ELI}}$ ELI (${{SESSION_SUMMARY.PC_ELI}}$\%) \\ \hline
\end{tabular}
\caption{An intrusion was a word that was vocalized during the retrieval period that was not studied on the most recent list. Intrusions were either words from a previous list (\textbf{PLI}: prior-list intrusions) or words that were not studied at all (\textbf{ELI}: extra-list intrusions).}
\end{table}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c}
\multicolumn{3}{c}{\textbf{Math distractor}} \\
\hline
${{SESSION_SUMMARY.N_MATH}}$ math problems & ${{SESSION_SUMMARY.N_CORRECT_MATH}}$ correct (${{SESSION_SUMMARY.PC_CORRECT_MATH}}$\%) & ${{SESSION_SUMMARY.MATH_PER_LIST}}$ problems per list  \\ \hline
\end{tabular}
\caption{After each list, the patient was given 20 seconds to perform as many arithmetic problems as possible, which served as a distractor before the beginning of recall.}
\end{table}

\begin{figure}[!h]
\centering
\textbf{Probability of Recall}

\subfigure{\includegraphics[scale=0.4]{{{SESSION_SUMMARY.PROB_RECALL_PLOT_FILE}}}}
\caption{\textbf{Free recall:}
(a) Overall probability of recall as a function of serial position.
(b) Probability of first recall as a function of serial position.}
\end{figure}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c|c}
\multicolumn{4}{c}{\textbf{Stim vs Non-Stim Recalls}} \\
\hline
${{SESSION_SUMMARY.N_CORRECT_STIM}}/{{SESSION_SUMMARY.N_TOTAL_STIM}}$ (${{SESSION_SUMMARY.PC_FROM_STIM}}$\%) from stim lists
 & ${{SESSION_SUMMARY.N_CORRECT_NONSTIM}}/{{SESSION_SUMMARY.N_TOTAL_NONSTIM}}$ (${{SESSION_SUMMARY.PC_FROM_NONSTIM}}$\%) from non-stim lists & $\chi^2(1)={{SESSION_SUMMARY.CHISQR}}$ & $p={{SESSION_SUMMARY.PVALUE}}$ \\
\hline
 {{SESSION_SUMMARY.ITEMLEVEL_COMPARISON}}
\end{tabular}
\end{table}

\begin{table}[!h]
\centering
\begin{tabular}{c|c}
\multicolumn{2}{c}{\textbf{Stim vs Non-Stim Intrusions}} \\
\hline
${{SESSION_SUMMARY.N_STIM_INTR}}/{{N_TOTAL_STIM}}$ (${{SESSION_SUMMARY.PC_FROM_STIM_INTR}}$\%) from stim lists &
${{SESSION_SUMMARY.N_NONSTIM_INTR}}/{{SESSION_SUMMARY.N_TOTAL_NONSTIM}}$ (${{SESSION_SUMMARY.PC_FROM_NONSTIM_INTR}}$\%) from non-stim lists \\
\hline
\end{tabular}
\end{table}
\clearpage

\begin{center}
\textbf{\Large Stim and recall analysis}
\end{center}

\begin{figure}[!h]
\centering
\subfigure{\includegraphics[height=0.25\textheight]{{{SESSION_SUMMARY.STIM_AND_RECALL_PLOT_FILE}}}}
\caption*{\textbf{Stim and recall:} Number of stims and recalled items per list. Red circles represent the number of recalled items from stim lists. Blue circles represent the number of recalled items from non-stim lists. Bars represent the number of stims per list.}
\textbf{Probability of Stimulation}

\subfigure{\includegraphics[scale=0.3]{{{SESSION_SUMMARY.PROB_STIM_PLOT_FILE}} !}}
\caption*{Probability of stimulation as a function of serial position.}
\end{figure}

{% if SESSION_SUMMARY.STIM_VS_NON_STIM_HALVES_PLOT_FILE %}
% \vspace{1pc}

\begin{figure}[!ht]
\centering
\includegraphics[height=0.25\textheight]{{{SESSION_SUMMARY.STIM_VS_NON_STIM_HALVES_PLOT_FILE}}}
\caption*{\textbf{(a)} Change in classifier output from the stimulated item (0-1366 ms) to the post-stimulated item (0-1366 ms) plotted for all items and for low-classifier output items. The control group comprises all items from non-stim lists.
\textbf{(b)} Change in recall performance for stimulated items contingent upon whether the stimulated item was presented during a low classifier state or a high classifier state (0-1366 ms).
\textbf{(c)} Change in recall performance for post-stimulated items contingent upon whether the stimulated item was presented during a low classifier state or a high classifier state (0-1366 ms).}
\end{figure}
{% end %}

{% end %}
{% endblock %}