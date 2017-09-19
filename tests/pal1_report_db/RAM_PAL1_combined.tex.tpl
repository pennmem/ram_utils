\documentclass[a4paper]{article} 
\usepackage[usenames,dvipsnames,svgnames,table]{xcolor}
\usepackage{graphicx,multirow} 
%\usepackage{epstopdf}
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

\addtolength{\oddsidemargin}{-.875in} 
\addtolength{\evensidemargin}{-.875in} 
\addtolength{\textwidth}{1.75in} 
\addtolength{\topmargin}{-.75in} 
\addtolength{\textheight}{1.75in} 

\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}} 
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\lhead{RAM PAL1 report v 2.2}
\chead{Subject: \textbf{<SUBJECT>}}
\rhead{Date created: <DATE>}
\begin{document}


\section*{<SUBJECT> RAM PAL1 Report}

\begin{tabular}{ccc}
\begin{minipage}[htbp]{200pt}
In the paired associates task, participants are presented with a list of pairs of words. Later, the participants are presented with a single word
and asked to recall the word it was paired with.
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
\begin{tabular}{c|c|c|c|c|c|c}
\textbf{Pairs}  & \textbf{Correct}  & \shortstack{\textbf{Wilson's}\\\textbf{interval}} & \textbf{Voc pass}  & \shortstack{\textbf{Nonvoc}\\\textbf{pass}}  & \textbf{PLI} & \textbf{ELI} \\
\hline
$<N_PAIRS>$ & $<N_CORRECT_PAIRS>$ ($<PC_CORRECT_PAIRS>$\%) & $<WILSON1>$-$<WILSON2>$\% & $<N_VOC_PASS>$ ($<PC_VOC_PASS>$\%) & $<N_NONVOC_PASS>$ ($<PC_NONVOC_PASS>$\%) & $<N_PLI>$ ($<PC_PLI>$\%) & $<N_ELI>$ ($<PC_ELI>$\%) \\
\hline
\end{tabular}
\caption{An intrusion is a vocalized word that was incorrect. Intrusions are either words from a previous list (\textbf{PLI}: prior-list intrusions) or words that were not studied at all (\textbf{ELI}: extra-list intrusions).}
\end{table}

%\begin{table}[!h]
%\centering
%\begin{tabular}{c|c|c|c}
%\multicolumn{4}{c}{\textbf{Free Recall}} \\
%\hline
%$<N_WORDS>$ words & $<N_CORRECT_WORDS>$ correct ($<PC_CORRECT_WORDS>$\%) &$<N_PLI>$ PLI ($<PC_PLI>$\%) &$<N_ELI>$ ELI ($<PC_ELI>$\%) \\ \hline
%\end{tabular}
%\caption{An intrusion was a word that was vocalized during the retrieval period that was not studied on the most recent list. Intrusions were either words from a previous list (\textbf{PLI}: prior-list intrusions) or words that were not studied at all (\textbf{ELI}: extra-list intrusions).}
%\end{table}

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
\caption{\textbf{Paired Associates:} (a) Probability of recall as a function of serial position. (b) Probability of recall as a function of study-test lag.}
\end{figure}

\clearpage

\begin{center}
\textbf{\large Significant Electrodes} \\
\begin{longtable}{C{.75cm} C{2cm} C{2.5cm} C{5.5cm} C{1.25cm} C{1.25cm}}
\hline
Type & Channel \# & Electrode Pair & Atlas Loc & \textit{p} & \textit{t}-stat \\
<SIGNIFICANT_ELECTRODES>
\hline
\caption{Subsequent memory effect HFA. High frequency activity (HFA, 70-200 Hz) was measured from 400ms to 3700ms of pair presentation interval. At each electrode, a t-test compared HFA for subsequently recalled vs subsequently forgotten items. \textbf{Surface Electrodes:} Red - significant positive effect (subsequently \textit{remembered} HFA $>$ subsequently \textit{forgotten} HFA). Blue - significant negative effect (subsequently \textit{remembered} HFA $<$ subsequently \textit{forgotten} HFA). Black - difference not significant. \textbf{Depth Electrodes:} All bipolar pairs shown in descending order of significance Bipolar pairs that exceed significance threshold ($p < .05$) are in boldface.}
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
