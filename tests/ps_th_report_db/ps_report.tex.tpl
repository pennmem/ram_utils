\documentclass[a4paper]{article} 

\usepackage[table]{xcolor}
\usepackage{graphicx}
\usepackage{grffile}
\usepackage{caption}
\usepackage[skip=0pt]{subcaption}
\usepackage{morefloats}

\addtolength{\oddsidemargin}{-.875in} 
\addtolength{\evensidemargin}{-.875in} 
\addtolength{\textwidth}{1.75in} 
\addtolength{\topmargin}{-.75in} 
\addtolength{\textheight}{1.75in} 

\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}} 
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\lhead{<EXPERIMENT>-TH1 report v 1.0}
\chead{Subject: \textbf{<SUBJECT>}}
\rhead{Date created: <DATE>}
\begin{document}

\section*{<SUBJECT> RAM <EXPERIMENT> Parameter Search Report for TH1}

\begin{minipage}{0.5\textwidth}
\begin{itemize}
  \item \textbf{Number of sessions:} <NUMBER_OF_SESSIONS>
  \item \textbf{Number of electrodes:} <NUMBER_OF_ELECTRODES>
\end{itemize}
\end{minipage}
\begin{minipage}{0.5\textwidth}
\begin{tabular}{|c|c|c|}
\hline Session \# & Date & Length (min) \\
<SESSION_DATA>
\end{tabular}

\end{minipage}

\vspace{3pc}

\begin{center}
\textbf{\Large Classifier performance} \\
\end{center}

\begin{figure}[!ht]
\centering
\includegraphics[scale=0.45]{<ROC_AND_TERC_PLOT_FILE>}
\caption{\textbf{(a)} ROC curve for the subject;
\textbf{(b)} Subject recall performance represented as
percentage devation from the (subject) mean, separated by tercile
of the classifier encoding efficiency estimate for each encoded word.}
\end{figure}

$\bullet$ Area Under Curve = $<AUC>$\%

$\bullet$ Permutation test $p$-value $<PERM-P-VALUE>$

<REPORT_PAGES>

\clearpage

\section*{APPENDIX: Calculating Expected Recall Change}

\[ \textrm{Expected Recall Change} = \left. \left( \frac{N_1 \Phi_1(\xi_{\textrm{post}})}{N_1 \Phi_1(\xi_{\textrm{post}}) + N_0 \Phi_0(\xi_{\textrm{post}})} - \frac{N_1 \Phi_1(\xi_{\textrm{pre}})}{N_1 \Phi_1(\xi_{\textrm{pre}}) + N_0 \Phi_0(\xi_{\textrm{pre}})}\right) \middle/ (N_1/N) \right., \]
$\bullet$ $N_1$ is \#recalls, $N_0$ is \#non-recalls, $N=N_1+N_0$; \\
$\bullet$ $\xi_{\textrm{post}} = \ln \frac{C_{\textrm{post}}}{1-C_{\textrm{post}}}$ is inverse logit of classifier post-stim output $C_{\textrm{post}}$; \\
$\bullet$ $\xi_{\textrm{pre}} = \ln \frac{C_{\textrm{pre}}}{1-C_{\textrm{pre}}}$ is inverse logit of classifier pre-stim output $C_{\textrm{pre}}$; \\
$\bullet$ $\Phi_1,\Phi_0$ are cdfs of Normal fits for inverse logit of classifier output for recalls/non-recalls with equal variance.

\end{document}
