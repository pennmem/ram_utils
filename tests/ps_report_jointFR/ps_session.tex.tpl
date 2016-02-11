
\section*{\hfil Session <SESS_NUM> \hfil}

\begin{tabular}{ccc}
\begin{minipage}[htbp]{160pt}
\textbf{Parameters:} \\
$\bullet$ <CONSTANT_NAME>: $<CONSTANT_VALUE>$ <CONSTANT_UNIT> \\
$\bullet$ ISI: $<ISI_MID>$ ($\pm <ISI_HALF_RANGE>$) ms \\
$\bullet$ Channel: <STIMTAG> \\
$\bullet$ Region: <REGION>
\end{minipage}
&
\begin{minipage}[htbp]{280pt}
\centering
\textbf{Two-factor ANOVA}
% {<PARAMETER1> $\times$ <PARAMETER2>}

\begin{tabular}{|c|c|c|c|}
\hline & <PARAMETER1> & <PARAMETER2> & <PARAMETER1> $\times$ <PARAMETER2> \\
\hline $F$ & $<FVALUE1>$ & $<FVALUE2>$ & $<FVALUE12>$ \\
\hline $p$ & $<PVALUE1>$ & $<PVALUE2>$ & $<PVALUE12>$ \\
\hline
\end{tabular}
\end{minipage}
\end{tabular}

\begin{figure}[!h]
\centering
\subfigure{\includegraphics[scale=0.35]{<PLOT_FILE>}}
\end{figure}

<PARAM1_TTEST_TABLE>
<PARAM2_TTEST_TABLE>
<PARAM12_TTEST_TABLE>

\clearpage
