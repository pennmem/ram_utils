\documentclass[a4paper]{article} 
\usepackage[usenames,dvipsnames,svgnames,table]{xcolor}
\usepackage{graphicx,multirow} 
\usepackage{epstopdf} 
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
\lhead{RAM FR1 \& CatFR1 report v 2.9}
\chead{Subject: \textbf{<SUBJECT>}}
\rhead{Date created: <DATE>}
\begin{document}


\section*{<SUBJECT> RAM FR1 \& CatFR1 Joint Report}

\begin{tabular}{ccc}
\begin{minipage}[htbp]{200pt}
In the free recall task, participants are presented with a list of words, one after the other, and later asked to recall as many words as possible in any order.
\begin{itemize}\item\textbf{Number of sessions: }$<NUMBER_OF_SESSIONS>$\item\textbf{Number of electrodes: }$<NUMBER_OF_ELECTRODES>$\end{itemize}
\end{minipage}
&
\begin{tabular}{|c|c|c|c|c|}
\hline Session \# & Date & Length (min) & \#lists & Perf \\
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
\includegraphics[scale=0.4]{<PROB_RECALL_PLOT_FILE>}
\caption{\textbf{Free recall:} (a) Overall probability of recall as a function of serial position. (b) Probability of FIRST recall as a function of serial position.}
\end{figure}

\clearpage

\begin{center}
\textbf{\large Significant Electrodes} \\
\begin{longtable}{<TABLE_FORMAT>}
\hline
<TABLE_HEADER> \\
<SIGNIFICANT_ELECTRODES>
\hline
\caption{Subsequent memory effect HFA. High frequency activity (HFA, 70-200 Hz) was measured across the word presentation interval (0 to 1600ms). At each electrode, a t-test compared HFA for subsequently recalled vs subsequently forgotten items. \textbf{Surface Electrodes:} Red - significant positive effect (subsequently \textit{remembered} HFA $>$ subsequently \textit{forgotten} HFA). Blue - significant negative effect (subsequently \textit{remembered} HFA $<$ subsequently \textit{forgotten} HFA). Black - difference not significant. \textbf{Depth Electrodes:} All bipolar pairs shown in descending order of significance Bipolar pairs that exceed significance threshold ($p < .05$) are in boldface.}
\end{longtable}
\end{center}

\clearpage
\begin{center}
\textbf{\Large Multivariate classification analysis}
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

$\bullet$ Median of classifier output = $<J-THRESH>$

%\begin{figure}[!h]
%\centering
%\includegraphics[width=0.45\textwidth]{/home1/busygin/scratch/reports/RAM_FR1/R1092J_2/analyses/RAM_FR1_SME_HFA_fft/012/ABSW_RAM_FR1_R1092J_2.eps}
%\caption{Average absolute classifier weights for broad regions of
%interest. All=all electrodes; FC=frontal cortex; PFC=prefrontal cortex;
%TC=temporal cortex; MTL=medial temporal lobe (including hippocampus,
%amygdala and cortex); HC=hippocampus; OC=occipital cortex; PC=parietal cortex.}
%\end{figure}


\end{document}
