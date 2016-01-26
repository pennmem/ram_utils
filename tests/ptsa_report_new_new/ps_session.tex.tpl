
\section*{\hfil Session <SESS_NUM> \hfil}

\begin{tabular}{ccc}
\begin{minipage}[htbp]{160pt}
\textbf{Parameters:}
\begin{itemize}
  \item <CONSTANT_NAME>: $<CONSTANT_VALUE>$ <CONSTANT_UNIT>
  \item ISI: $<ISI_MID>$ ($\pm <ISI_HALF_RANGE>$) ms
  \item Channel: <STIMTAG>
\end{itemize}
\end{minipage}
&
\begin{minipage}[htbp]{280pt}
\textbf{Two-factor ANOVA for <PARAMETER1> $\times$ <PARAMETER2>}

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
\subfigure{\includegraphics[scale=0.45]{<PLOT_FILE>}}
%\caption{Left: <PARAMETER1> vs classifier delta. Right: <PARAMETER2> vs classifier delta. All trials: (a,b). Lowest tercile: (c,d). Highest tercile: (e,f).}
\end{figure}

\clearpage
