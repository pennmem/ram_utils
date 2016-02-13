\documentclass[a4paper]{article}

\usepackage{graphicx,multirow}
\usepackage{subfigure,amsmath}

\addtolength{\oddsidemargin}{-.675in} 
\addtolength{\evensidemargin}{-.675in} 
\addtolength{\textwidth}{1.7in} 
\addtolength{\topmargin}{-.55in} 
\addtolength{\textheight}{1.55in} 

\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\lhead{PS Aggregate Report v 1.0}
\chead{PS1,PS2}
\rhead{Date created: <DATE>}
\begin{document}

\section*{\hfil PS1,PS2 Region $\times$ Pulse Frequency Analysis \hfil}

%\textbf{\hfil Two-factor ANOVA \hfil}
%
%\begin{tabular}{|c|c|c|c|}
%\hline & Region & Frequency & Region $\times$ Frequency \\
%\hline $F$ & $<FVALUERF1>$ & $<FVALUERF2>$ & $<FVALUERF12>$ \\
%\hline $p$ & $<PVALUERF1>$ & $<PVALUERF2>$ & $<PVALUERF12>$ \\
%\hline
%\end{tabular}

\begin{figure}[!h]
\centering
\subfigure{\includegraphics[scale=0.4]{<FREQUENCY_PLOT_FILE>}}
\end{figure}

\textbf{\hfil Experiment count \hfil}

\begin{tabular}{|l|c|c|c|c|c|c|}
<REGION_FREQUENCY_EXPERIMENT_COUNT_TABLE>
\end{tabular}

\clearpage

\section*{\hfil PS2 Region $\times$ Amplitude Analysis ($10$ and $25$ Hz) \hfil}

%\textbf{\hfil Two-factor ANOVA \hfil}
%
%\begin{tabular}{|c|c|c|c|}
%\hline & Region & Amplitude & Region $\times$ Amplitude \\
%\hline $F$ & $<FVALUERALOW1>$ & $<FVALUERALOW2>$ & $<FVALUERALOW12>$ \\
%\hline $p$ & $<PVALUERALOW1>$ & $<PVALUERALOW2>$ & $<PVALUERALOW12>$ \\
%\hline
%\end{tabular}

\begin{figure}[!h]
\centering
\subfigure{\includegraphics[scale=0.4]{<AMPLITUDE_LOW_PLOT_FILE>}}
\end{figure}

\clearpage

\section*{\hfil PS2 Region $\times$ Amplitude Analysis ($100$ and $200$ Hz) \hfil}

%\textbf{\hfil Two-factor ANOVA \hfil}
%
%\begin{tabular}{|c|c|c|c|}
%\hline & Region & Amplitude & Region $\times$ Amplitude \\
%\hline $F$ & $<FVALUERAHIGH1>$ & $<FVALUERAHIGH2>$ & $<FVALUERAHIGH12>$ \\
%\hline $p$ & $<PVALUERAHIGH1>$ & $<PVALUERAHIGH2>$ & $<PVALUERAHIGH12>$ \\
%\hline
%\end{tabular}

\begin{figure}[!h]
\centering
\subfigure{\includegraphics[scale=0.4]{<AMPLITUDE_HIGH_PLOT_FILE>}}
\end{figure}

\clearpage

\section*{\hfil PS1 Region $\times$ Duration Analysis ($10$ and $25$ Hz) \hfil}

%\textbf{\hfil Two-factor ANOVA \hfil}
%
%\begin{tabular}{|c|c|c|c|}
%\hline & Region & Duration & Region $\times$ Duration \\
%\hline $F$ & $<FVALUERDLOW1>$ & $<FVALUERDLOW2>$ & $<FVALUERDLOW12>$ \\
%\hline $p$ & $<PVALUERDLOW1>$ & $<PVALUERDLOW2>$ & $<PVALUERDLOW12>$ \\
%\hline
%\end{tabular}

\begin{figure}[!h]
\centering
\subfigure{\includegraphics[scale=0.4]{<DURATION_LOW_PLOT_FILE>}}
\end{figure}

\clearpage

\section*{\hfil PS1 Region $\times$ Duration Analysis ($100$ and $200$ Hz) \hfil}

%\textbf{\hfil Two-factor ANOVA \hfil}
%
%\begin{tabular}{|c|c|c|c|}
%\hline & Region & Duration & Region $\times$ Duration \\
%\hline $F$ & $<FVALUERDHIGH1>$ & $<FVALUERDHIGH2>$ & $<FVALUERDHIGH12>$ \\
%\hline $p$ & $<PVALUERDHIGH1>$ & $<PVALUERDHIGH2>$ & $<PVALUERDHIGH12>$ \\
%\hline
%\end{tabular}

\begin{figure}[!h]
\centering
\subfigure{\includegraphics[scale=0.4]{<DURATION_HIGH_PLOT_FILE>}}
\end{figure}

\clearpage

\section*{\hfil Number of collected PS1 \& PS2 sessions per region \hfil}

\begin{tabular}{|l|r|}
\hline Region & \# of sessions \\
\hline <REGION_SESSION_TOTAL_DATA>
\hline
\end{tabular}


\end{document}
