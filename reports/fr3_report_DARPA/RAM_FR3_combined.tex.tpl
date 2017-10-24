\documentclass[a4paper]{article} 
\usepackage[usenames,dvipsnames,svgnames,table]{xcolor}
\usepackage{graphicx,multirow} 
\usepackage{epstopdf} 
\usepackage{subfigure,amsmath} 
\usepackage{wrapfig} 
\usepackage{longtable} 
\usepackage{pdfpages}
\usepackage{mathtools}
\usepackage{array}
\usepackage{enumitem}
\usepackage{booktabs}
\usepackage{sidecap} \usepackage{soul}
\usepackage[small,bf,it]{caption}
\setlength\belowcaptionskip{2pt}

\addtolength{\oddsidemargin}{-.875in} 
\addtolength{\evensidemargin}{-.875in} 
\addtolength{\textwidth}{1.75in} 
\addtolength{\topmargin}{-.75in} 
\addtolength{\textheight}{1.75in} 

\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}} 
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\lhead{RAM FR3 report v 1.0}
\rhead{Date created: <DATE>}
\begin{document}


\section*{<SUBJECT> RAM FR3 Free Recall Report}

\begin{tabular}{ccc}
\begin{minipage}[htbp]{250pt}
In the free recall task, participants are presented with a list of words, one after the other, and later asked to recall as many words as possible in any order.
\begin{itemize}\item\textbf{Number of sessions: }$<NUMBER_OF_SESSIONS>$\item\textbf{Number of electrodes: }$<NUMBER_OF_ELECTRODES>$\end{itemize}
\end{minipage}
&
\begin{tabular}{|c|c|c|}
\hline Session \# & Date & Length (min)\\
<SESSION_DATA>
\end{tabular}
\end{tabular}
\vspace{0.3 in}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c|c}
\multicolumn{4}{c}{\textbf{Free Recall}} \\ 
\hline
$<N_WORDS>$ words & $<N_CORRECT_WORDS>$ correct ($<PC_CORRECT_WORDS>$\%) &$<N_PLI>$ PLI ($<PC_PLI>$\%) &$<N_ELI>$ ELI ($<PC_ELI>$\%) \\ \hline 
\end{tabular}
\caption{An intrusion was a word that was vocalized during the retrieval period that was not studied on the most recent list. Intrusions were either words from a previous list (\textbf{PLI}: prior-list intrusions) or words that were not studied at all (\textbf{ELI}: extra-list intrusions).}
\end{table}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c}
\multicolumn{3}{c}{\textbf{Math distractor}} \\ 
\hline
$<N_MATH>$ math problems & $<N_CORRECT_MATH>$ correct ($<PC_CORRECT_MATH>$\%) & $<MATH_PER_LIST>$ problems per list  \\ \hline 
\end{tabular}
\caption{After each list, the patient was given 20 seconds to perform as many arithmetic problems as possible, which served as a distractor before the beginning of recall.}
\end{table}

\begin{figure}[!h]
\centering
\subfigure[]{\includegraphics[scale=0.4]{<PROB_RECALL_PLOT_FILE>}}
\caption{\textbf{Free recall:} (a) Overall probability of recall as a function of serial position. (b) Probability of FIRST recall as a function of serial position.}
\end{figure}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c|c}
\multicolumn{4}{c}{\textbf{Stim vs Non-Stim Recalls}} \\ 
\hline
$<N_CORRECT_STIM>/<N_TOTAL_STIM>$ ($<PC_FROM_STIM>$\%) from stim lists & $<N_CORRECT_NONSTIM>/<N_TOTAL_NONSTIM>$ ($<PC_FROM_NONSTIM>$\%) from non-stim lists & $\chi^2(1)=<CHISQR>$ & $p=<PVALUE>$ \\
\hline 
\end{tabular}
\end{table}

\begin{table}[!h]
\centering
\begin{tabular}{c|c}
\multicolumn{2}{c}{\textbf{Stim vs Non-Stim Intrusions}} \\ 
\hline
$<N_STIM_INTR>/<N_TOTAL_STIM>$ ($<PC_FROM_STIM_INTR>$\%) from stim lists & $<N_NONSTIM_INTR>/<N_TOTAL_NONSTIM>$ ($<PC_FROM_NONSTIM_INTR>$\%) from non-stim lists \\
\hline 
\end{tabular}
\end{table}

\end{document}
