\documentclass[a4paper]{article}

\usepackage{graphicx,multirow}
\usepackage{subfigure,amsmath}

\addtolength{\oddsidemargin}{-.875in}
\addtolength{\evensidemargin}{-.875in}
\addtolength{\textwidth}{1.75in}
\addtolength{\topmargin}{-.75in}
\addtolength{\textheight}{1.75in}

\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\lhead{PS Aggregate Report v 1.1}
\chead{PS1,PS2}
\rhead{Date created: <DATE>}
\begin{document}

\section*{\hfil PS1,PS2 Region $\times$ Pulse Frequency Analysis \hfil}

\begin{figure}[!h]
\centering
\subfigure{\includegraphics[scale=0.4]{<LOW_QUANTILE_FREQUENCY_PLOT_FILE>}}
\subfigure{\includegraphics[scale=0.4]{<HIGH_QUANTILE_FREQUENCY_PLOT_FILE>}}
\subfigure{\includegraphics[scale=0.4]{<ALL_FREQUENCY_PLOT_FILE>}}
\end{figure}

\clearpage

\section*{\hfil PS2 Region $\times$ Amplitude Analysis \hfil}

\begin{figure}[!h]
\centering
\subfigure{\includegraphics[scale=0.4]{<LOW_QUANTILE_AMPLITUDE_PLOT_FILE>}}
\subfigure{\includegraphics[scale=0.4]{<HIGH_QUANTILE_AMPLITUDE_PLOT_FILE>}}
\subfigure{\includegraphics[scale=0.4]{<ALL_AMPLITUDE_PLOT_FILE>}}
\end{figure}

\clearpage

\section*{\hfil PS1 Region $\times$ Duration Analysis \hfil}

\begin{figure}[!h]
\centering
\subfigure{\includegraphics[scale=0.4]{<LOW_QUANTILE_DURATION_PLOT_FILE>}}
\subfigure{\includegraphics[scale=0.4]{<HIGH_QUANTILE_DURATION_PLOT_FILE>}}
\subfigure{\includegraphics[scale=0.4]{<ALL_DURATION_PLOT_FILE>}}
\end{figure}

\clearpage

\section*{\hfil Number of collected PS1 \& PS2 sessions per region \hfil}

\textbf{\hfil Experiment count \hfil}

\begin{tabular}{|l|c|c|c|c|c|c|}
<REGION_FREQUENCY_EXPERIMENT_COUNT_TABLE>
\end{tabular}


\begin{tabular}{|l|r|}
\hline Region & \# of sessions \\
\hline <REGION_SESSION_TOTAL_DATA>
\hline
\end{tabular}


\end{document}
