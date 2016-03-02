\documentclass[a4paper]{article}

\usepackage{graphicx}
\usepackage{caption}
\setlength{\abovecaptionskip}{2pt}

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
\chead{PS3}
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

\subsection*{\hfil Anatomical Category $\times$ Burst Frequency Experiment Count \hfil}

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

\subsection*{\hfil PS3 Region $\times$ Burst Frequency Analysis \hfil}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.4]{<ALL_FREQUENCY_CLASSIFIER_DELTA_PLOT_FILE>}
\caption{Cumulative means and standard errors of difference in classifier output post- and pre-stim}
\end{figure}

\vspace{1cm}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.4]{<ALL_FREQUENCY_RECALL_CHANGE_PLOT_FILE>}
\caption{Cumulative means and standard errors of:}
\[ \textrm{Expected Recall Change} = \frac{N_{recalls}(C \leq C_{\textrm{post}})/N_{items}(C \leq C_{\textrm{post}}) - N_{recalls}(C \leq C_{\textrm{pre}})/N_{items}(C \leq C_{\textrm{pre}})}{N_{recalls}/N_{items}}, \]
\raggedright
where $C_{\textrm{pre}}$ is pre-stim classifier output, $C_{\textrm{post}}$ is post-stim classifier output.
\end{figure}

\clearpage

\subsection*{\hfil PS3 Region $\times$ Burst Frequency Analysis (100 Hz) \hfil}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.4]{<ALL_100_FREQUENCY_CLASSIFIER_DELTA_PLOT_FILE>}
\caption{Cumulative means and standard errors of difference in classifier output post- and pre-stim}
\end{figure}

\vspace{1cm}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.4]{<ALL_100_FREQUENCY_RECALL_CHANGE_PLOT_FILE>}
\caption{Cumulative means and standard errors of:}
\[ \textrm{Expected Recall Change} = \frac{N_{recalls}(C \leq C_{\textrm{post}})/N_{items}(C \leq C_{\textrm{post}}) - N_{recalls}(C \leq C_{\textrm{pre}})/N_{items}(C \leq C_{\textrm{pre}})}{N_{recalls}/N_{items}}, \]
\raggedright
where $C_{\textrm{pre}}$ is pre-stim classifier output, $C_{\textrm{post}}$ is post-stim classifier output.
\end{figure}

\clearpage

\subsection*{\hfil PS3 Region $\times$ Burst Frequency Analysis (200 Hz) \hfil}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.4]{<ALL_200_FREQUENCY_CLASSIFIER_DELTA_PLOT_FILE>}
\caption{Cumulative means and standard errors of difference in classifier output post- and pre-stim}
\end{figure}

\vspace{1cm}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.4]{<ALL_200_FREQUENCY_RECALL_CHANGE_PLOT_FILE>}
\caption{Cumulative means and standard errors of:}
\[ \textrm{Expected Recall Change} = \frac{N_{recalls}(C \leq C_{\textrm{post}})/N_{items}(C \leq C_{\textrm{post}}) - N_{recalls}(C \leq C_{\textrm{pre}})/N_{items}(C \leq C_{\textrm{pre}})}{N_{recalls}/N_{items}}, \]
\raggedright
where $C_{\textrm{pre}}$ is pre-stim classifier output, $C_{\textrm{post}}$ is post-stim classifier output.
\end{figure}

\end{document}
