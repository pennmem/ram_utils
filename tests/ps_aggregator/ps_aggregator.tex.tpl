\documentclass[a4paper]{article}

\usepackage{graphicx}
\usepackage{caption}
\setlength{\abovecaptionskip}{0pt}
\usepackage[skip=0pt]{subcaption}

\usepackage{color}

\addtolength{\oddsidemargin}{-.875in}
\addtolength{\evensidemargin}{-.875in}
\addtolength{\textwidth}{1.75in}
\addtolength{\topmargin}{-.75in}
\addtolength{\textheight}{1.75in}

\setlength{\textfloatsep}{1pt plus 1.0pt minus 1.0pt}
\setlength{\floatsep}{1pt plus 1.0pt minus 1.0pt}
\setlength{\intextsep}{1pt plus 1.0pt minus 1.0pt}


\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\lhead{PS Aggregate Report v 2.0}
\chead{PS1,PS2}
\rhead{Date created: <DATE>}
\begin{document}

\subsection*{\hfil Anatomical Categories \hfil}

\begin{table}[!h]
\centering
\begin{tabular}{|c|c|c|}
\hline
HC & MTLC & Cing-PFC \\
\hline
CA1, CA2, CA3, DG, Sub & PRC, PHC, EC & PCg, ACg, DLPFC \\
\hline
\end{tabular}
\end{table}

\subsection*{\hfil Anatomical Category $\times$ Frequency Experiment Count \hfil}

\begin{table}[!h]
\centering
\begin{tabular}{|l|c|c|c|c|c|c|}
<REGION_FREQUENCY_EXPERIMENT_COUNT_TABLE>
\end{tabular}
\end{table}

\subsection*{\hfil Experiment Count Per Region \hfil}

\begin{table}[!h]
\centering
\begin{tabular}{|l|r|}
\hline Region & \# of sessions \\
\hline <REGION_SESSION_TOTAL_DATA>
\hline
\end{tabular}
\end{table}

\clearpage

\subsection*{\hfil PS1,PS2 Region $\times$ Pulse Frequency Analysis \hfil}
%{\color{blue} Blue: HC}; {\color{green} Green: MTLC}; {\color{red} Red: Cing-PFC}; {\color{cyan} Cyan: Other}

\begin{figure}[!h]
\centering
\begin{subfigure}[!h]{\linewidth}
\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{<LOW_QUANTILE_FREQUENCY_PLOT_FILE>}
\caption{Lower Half of Pre-Stim Classifier Output}
\end{subfigure}
\vspace{10pt}
\begin{subfigure}[!h]{\linewidth}
\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{<HIGH_QUANTILE_FREQUENCY_PLOT_FILE>}
\caption{Upper Half of Pre-Stim Classifier Output}
\end{subfigure}
\begin{subfigure}[!h]{\linewidth}
\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{<ALL_FREQUENCY_PLOT_FILE>}
\caption{All Trials}
\end{subfigure}
\end{figure}

\clearpage

\subsection*{\hfil PS2 Region $\times$ Amplitude Analysis \hfil}

\hspace{2cm} \textbf{Low Frequencies (10 and 25 Hz)} \hspace{1.5cm} \textbf{High Frequencies (100 and 200 Hz)}

\begin{figure}[!h]
\centering
\begin{subfigure}[!h]{\linewidth}
\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{<LOW_QUANTILE_AMPLITUDE_PLOT_FILE>}
\subcaption{Lower Half of Pre-Stim Classifier Output}
\end{subfigure}
\begin{subfigure}[!h]{\linewidth}
\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{<HIGH_QUANTILE_AMPLITUDE_PLOT_FILE>}
\subcaption{Upper Half of Pre-Stim Classifier Output}
\end{subfigure}
\begin{subfigure}[!h]{\linewidth}
\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{<ALL_AMPLITUDE_PLOT_FILE>}
\subcaption{All Trials}
\end{subfigure}
\end{figure}

\clearpage

\subsection*{\hfil PS1 Region $\times$ Duration Analysis \hfil}

\hspace{2cm} \textbf{Low Frequencies (10 and 25 Hz)} \hspace{1.5cm} \textbf{High Frequencies (100 and 200 Hz)}

\begin{figure}[!h]
\centering
\begin{subfigure}[!h]{\linewidth}
\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{<LOW_QUANTILE_DURATION_PLOT_FILE>}
\subcaption{Lower Half of Pre-Stim Classifier Output}
\end{subfigure}
\begin{subfigure}[!h]{\linewidth}
\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{<HIGH_QUANTILE_DURATION_PLOT_FILE>}
\subcaption{Upper Half of Pre-Stim Classifier Output}
\end{subfigure}
\begin{subfigure}[!h]{\linewidth}
\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{<ALL_DURATION_PLOT_FILE>}
\subcaption{All Trials}
\end{subfigure}
\end{figure}


\end{document}
