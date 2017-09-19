\documentclass[a4paper]{article}
\usepackage[usenames,dvipsnames,svgnames,table]{xcolor}
\usepackage{graphicx,multirow}
%\usepackage{epstopdf}
\usepackage{subfigure,amsmath}
%\usepackage{wrapfig}
\usepackage{longtable}
\usepackage{pdfpages}
\usepackage{mathtools}
\usepackage{array}
\usepackage{enumitem}
\usepackage{booktabs}
%\usepackage{sidecap} \usepackage{soul}
\usepackage[small,bf,it]{caption}
\setlength\belowcaptionskip{2pt}

\addtolength{\oddsidemargin}{-1.0in}
\addtolength{\evensidemargin}{-1.0in}
\addtolength{\textwidth}{2.0in}
\addtolength{\topmargin}{-.75in}
\addtolength{\textheight}{1.75in}

\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\lhead{RAM <EXPERIMENT> report v 2.3}
\chead{Subject: \textbf{<SUBJECT>}}
\rhead{Date created: <DATE>}
\begin{document}


\section*{<SUBJECT> RAM <EXPERIMENT> Report}

\begin{tabular}{ccc}
\begin{minipage}[htbp]{170pt}
In the treasure hunt task, participants navigate a virtual environment and encounter objects hidden in treasure chests. Participants are asked to remember the object location by placing a target where they believe the object was originally located.
\begin{itemize}\item\textbf{Number of sessions: }$<NUMBER_OF_SESSIONS>$\item\textbf{Number of electrodes: }$<NUMBER_OF_ELECTRODES>$\end{itemize}
\end{minipage}
&
\begin{tabular}{|c|c|c|c|c|c|}
\hline Session & Date & Length (min) & \#lists & Perf & Amp (mA) \\
<SESSION_DATA>
\end{tabular}
\end{tabular}

\vspace{3pc}

\begin{center}
\textbf{\Large TH1 Classifier performance}
\end{center}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.45]{<ROC_AND_TERC_PLOT_FILE>}
\caption{\textbf{(a)} ROC curve for the subject;
\textbf{(b)} Subject recall performance represented as
percentage devation from the (subject) mean, separated by tercile
of the classifier encoding efficiency estimate for each encoded word.}
\end{figure}

$\bullet$ Number of TH1 sessions = $<NUMBER_TH1_SESSIONS>$

$\bullet$ Area Under Curve = $<AUC>$\%

$\bullet$ Permutation test $p$-value $<PERM-P-VALUE>$

<REPORT_PAGES>

\end{document}
