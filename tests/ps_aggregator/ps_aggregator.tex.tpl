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
%\[ \textrm{Expected Recall Change} = \frac{N_{recalls}(C \leq C_{\textrm{post}})/N_{items}(C \leq C_{\textrm{post}}) - N_{recalls}(C \leq C_{\textrm{pre}})/N_{items}(C \leq C_{\textrm{pre}})}{N_{recalls}/N_{items}}, \]
%\raggedright
%where $C_{\textrm{pre}}$ is pre-stim classifier output, $C_{\textrm{post}}$ is post-stim classifier output.
\end{figure}
\[ \textrm{Expected Recall Change} = \left. \left( \frac{N_1 \Phi_1(\xi_{\textrm{post}})}{N_1 \Phi_1(\xi_{\textrm{post}}) + N_0 \Phi_0(\xi_{\textrm{post}})} - \frac{N_1 \Phi_1(\xi_{\textrm{pre}})}{N_1 \Phi_1(\xi_{\textrm{pre}}) + N_0 \Phi_0(\xi_{\textrm{pre}})}\right) \middle/ (N_1/N) \right., \]
$\bullet$ $N_1$ is \#recalls, $N_0$ is \#non-recalls, $N=N_1+N_0$; \\
$\bullet$ $\xi_{\textrm{post}} = \ln \frac{C_{\textrm{post}}}{1-C_{\textrm{post}}}$ is inverse logit of classifier post-stim output $C_{\textrm{post}}$; \\
$\bullet$ $\xi_{\textrm{pre}} = \ln \frac{C_{\textrm{pre}}}{1-C_{\textrm{pre}}}$ is inverse logit of classifier pre-stim output $C_{\textrm{pre}}$; \\
$\bullet$ $\Phi_1,\Phi_0$ are cdfs of Normal fits for inverse logit of classifier output for recalls/non-recalls with equal variance.

\clearpage

\subsection*{\hfil PS2 Region $\times$ Amplitude Analysis \hfil}

\hspace{2cm} \textbf{Low Frequencies (10 and 25 Hz)} \hspace{1.5cm} \textbf{High Frequencies (100 and 200 Hz)}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.4]{<ALL_AMPLITUDE_CLASSIFIER_DELTA_PLOT_FILE>}
\caption{Cumulative means and standard errors of difference in classifier output post- and pre-stim}
\end{figure}

\vspace{1cm}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.4]{<ALL_AMPLITUDE_RECALL_CHANGE_PLOT_FILE>}
\caption{Cumulative means and standard errors of:}
%\[ \textrm{Expected Recall Change} = \frac{N_{recalls}(C \leq C_{\textrm{post}})/N_{items}(C \leq C_{\textrm{post}}) - N_{recalls}(C \leq C_{\textrm{pre}})/N_{items}(C \leq C_{\textrm{pre}})}{N_{recalls}/N_{items}}, \]
%\raggedright
%where $C_{\textrm{pre}}$ is pre-stim classifier output, $C_{\textrm{post}}$ is post-stim classifier output.
\end{figure}
\[ \textrm{Expected Recall Change} = \left. \left( \frac{N_1 \Phi_1(\xi_{\textrm{post}})}{N_1 \Phi_1(\xi_{\textrm{post}}) + N_0 \Phi_0(\xi_{\textrm{post}})} - \frac{N_1 \Phi_1(\xi_{\textrm{pre}})}{N_1 \Phi_1(\xi_{\textrm{pre}}) + N_0 \Phi_0(\xi_{\textrm{pre}})}\right) \middle/ (N_1/N) \right., \]
$\bullet$ $N_1$ is \#recalls, $N_0$ is \#non-recalls, $N=N_1+N_0$; \\
$\bullet$ $\xi_{\textrm{post}} = \ln \frac{C_{\textrm{post}}}{1-C_{\textrm{post}}}$ is inverse logit of classifier post-stim output $C_{\textrm{post}}$; \\
$\bullet$ $\xi_{\textrm{pre}} = \ln \frac{C_{\textrm{pre}}}{1-C_{\textrm{pre}}}$ is inverse logit of classifier pre-stim output $C_{\textrm{pre}}$; \\
$\bullet$ $\Phi_1,\Phi_0$ are cdfs of Normal fits for inverse logit of classifier output for recalls/non-recalls with equal variance.

\clearpage

\subsection*{\hfil PS1 Region $\times$ Duration Analysis \hfil}

\hspace{2cm} \textbf{Low Frequencies (10 and 25 Hz)} \hspace{1.5cm} \textbf{High Frequencies (100 and 200 Hz)}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.4]{<ALL_DURATION_CLASSIFIER_DELTA_PLOT_FILE>}
\caption{Cumulative means and standard errors of difference in classifier output post- and pre-stim}
\end{figure}

\vspace{1cm}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.4]{<ALL_DURATION_RECALL_CHANGE_PLOT_FILE>}
\caption{Cumulative means and standard errors of:}
%\[ \textrm{Expected Recall Change} = \frac{N_{recalls}(C \leq C_{\textrm{post}})/N_{items}(C \leq C_{\textrm{post}}) - N_{recalls}(C \leq C_{\textrm{pre}})/N_{items}(C \leq C_{\textrm{pre}})}{N_{recalls}/N_{items}}, \]
%\raggedright
%where $C_{\textrm{pre}}$ is pre-stim classifier output, $C_{\textrm{post}}$ is post-stim classifier output.
\end{figure}
\[ \textrm{Expected Recall Change} = \left. \left( \frac{N_1 \Phi_1(\xi_{\textrm{post}})}{N_1 \Phi_1(\xi_{\textrm{post}}) + N_0 \Phi_0(\xi_{\textrm{post}})} - \frac{N_1 \Phi_1(\xi_{\textrm{pre}})}{N_1 \Phi_1(\xi_{\textrm{pre}}) + N_0 \Phi_0(\xi_{\textrm{pre}})}\right) \middle/ (N_1/N) \right., \]
$\bullet$ $N_1$ is \#recalls, $N_0$ is \#non-recalls, $N=N_1+N_0$; \\
$\bullet$ $\xi_{\textrm{post}} = \ln \frac{C_{\textrm{post}}}{1-C_{\textrm{post}}}$ is inverse logit of classifier post-stim output $C_{\textrm{post}}$; \\
$\bullet$ $\xi_{\textrm{pre}} = \ln \frac{C_{\textrm{pre}}}{1-C_{\textrm{pre}}}$ is inverse logit of classifier pre-stim output $C_{\textrm{pre}}$; \\
$\bullet$ $\Phi_1,\Phi_0$ are cdfs of Normal fits for inverse logit of classifier output for recalls/non-recalls with equal variance.


\end{document}
