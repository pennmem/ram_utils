\documentclass[a4paper]{article} 

\usepackage[table]{xcolor}
\usepackage{graphicx}
\usepackage{grffile}
%\usepackage[skip=0pt]{subcaption}
%\setlength\belowcaptionskip{2pt}

\addtolength{\oddsidemargin}{-.875in} 
\addtolength{\evensidemargin}{-.875in} 
\addtolength{\textwidth}{1.75in} 
\addtolength{\topmargin}{-.75in} 
\addtolength{\textheight}{1.75in} 

\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}} 
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\lhead{<EXPERIMENT> report v 3.6}
\chead{Subject: \textbf{<SUBJECT>}}
\rhead{Date created: <DATE>}
\begin{document}

\section*{<SUBJECT> RAM <EXPERIMENT> Parameter Search Report}

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
\textbf{\Large Classifier performance}
\end{center}

\begin{figure}[!h]
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

\end{document}
