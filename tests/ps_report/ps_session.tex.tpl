
\clearpage

\section*{\hfil Channel <STIMTAG> (<REGION>) \hfil}

\begin{tabular}{ccc}
\begin{minipage}[htbp]{160pt}
\begin{itemize}
\item Session(s): <SESSIONS>
\item <CONSTANT_NAME>: $<CONSTANT_VALUE>$ <CONSTANT_UNIT>
\item ISI: $<ISI_MID>$ ($\pm <ISI_HALF_RANGE>$) ms
\end{itemize}
\end{minipage}
&
\begin{minipage}[htbp]{280pt}
\centering
\textbf{Two-factor ANOVA}

\begin{tabular}{|c|c|c|c|}
\hline & <PARAMETER1> & <PARAMETER2> & <PARAMETER1> $\times$ <PARAMETER2> \\
\hline $F$ & $<FVALUE1>$ & $<FVALUE2>$ & $<FVALUE12>$ \\
\hline $p$ & $<PVALUE1>$ & $<PVALUE2>$ & $<PVALUE12>$ \\
\hline
\end{tabular}
\end{minipage}
\end{tabular}

\begin{figure}[ht!]
<PS_PLOTS>
\caption{\textbf{(a)} Means and standard errors of difference in classifier output post- and pre-stim;
\textbf{(b)} Means and standard errors of {\em Expected Recall Change}.}
\end{figure}
<TTEST_AGAINST_SHAM_TABLE>
<TTEST_AGAINST_ZERO_TABLE>
