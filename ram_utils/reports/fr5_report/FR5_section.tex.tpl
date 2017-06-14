\section*{<SUBJECT> RAM FR5 Free Recall Report}

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

\vspace{3pc}

\begin{center}
\textbf{\Large <ROC_TITLE>}
\end{center}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.45]{<ROC_AND_TERC_PLOT_FILE>}
\caption{\textbf{(a)} ROC curve for the subject;
\textbf{(b)} Subject recall performance represented as
percentage devation from the (subject) mean, separated by tercile
of the classifier encoding efficiency estimate for each encoded word.}
\end{figure}

$\bullet$ Area Under Curve = $<FR5-AUC>$

$\bullet$ Permutation test $p$-value = $<FR5-PERM-P-VALUE>$

$\bullet$ Median of classifier output = $<FR5-JSTAT-THRESH>$

<REPORT_PAGES>

\end{document}
