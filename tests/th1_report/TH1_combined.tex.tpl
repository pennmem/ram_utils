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
\lhead{RAM TH1 report v 2.2}
\chead{Subject: \textbf{<SUBJECT>}}
\rhead{Date created: <DATE>}
\begin{document}


\section*{<SUBJECT> RAM TH1 Report}

\begin{tabular}{ccc}
\begin{minipage}[htbp]{175pt}
In the treasure hunt task, participants navigate a virtual environment and encounter objects hidden in treasure chests. Participants are asked to remember the object location by placing a target where they believe the object was originally located.
\begin{itemize}\item\textbf{Number of sessions: }$<NUMBER_OF_SESSIONS>$\item\textbf{Number of electrodes: }$<NUMBER_OF_ELECTRODES>$\end{itemize}
\end{minipage}
&
\small
\begin{tabular}{|c|c|c|c|c|c|}
    
\hline Session \# & Date & Length (m) & \#lists & Perf & Score\\
<SESSION_DATA>
\end{tabular}
\end{tabular}
\vspace{0.3 in}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c|c}
\multicolumn{4}{c}{\textbf{Treasure Hunt}} \\ 
\hline
$<N_ITEMS>$ items & $<N_CORRECT_ITEMS>$ correct & $<N_TRANSPOSE_ITEMS>$ correct counting transposed & $<MEAN_NORM_ERROR>$ mean normalized error\\ \hline
\end{tabular}
\caption{Items are considered correctly recalled if they are within the target circle. Normalized error of 0.5 indicates chance performance.}
\end{table}

% \begin{table}[!h]
% \centering
% \begin{tabular}{c|c|c}
% \multicolumn{3}{c}{\textbf{Math distractor}} \\
% \hline
% $<N_MATH>$ math problems & $<N_CORRECT_MATH>$ correct ($<PC_CORRECT_MATH>$\%) & $<MATH_PER_LIST>$ problems per list  \\ \hline
% \end{tabular}
% \caption{After each list, the patient was given 20 seconds to perform as many arithmetic problems as possible, which served as a distractor before the beginning of recall.}
% \end{table}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.4]{<PROB_RECALL_PLOT_FILE>}
\includegraphics[scale=0.4]{<DIST_HIST_PLOT_FILE>}
\includegraphics[scale=0.4]{<ERR_BLOCK_PLOT_FILE>}
\caption{\textbf{Treasure Hunt:} \emph{Top Left:} Probability of a correct retrieval as a function of confidence level. \emph{Top Right:} Histogram of distance error, plotted separately for each confidence level. \emph{Bottom:} Distance error as a function of block number.}
\end{figure}

\clearpage

\begin{center} 
\textbf{\large Significant Electrodes Low Theta} \\
\begin{longtable}{C{.75cm} C{2cm} C{2.5cm} C{5.5cm} C{1.25cm} C{1.25cm}}
\hline
Type & Channel \# & Electrode Pair & Atlas Loc & \textit{p} & \textit{t}-stat \\
<SIGNIFICANT_ELECTRODES_LTA>
\hline
\caption{Subsequent memory effect low theta. Low theta activity (1-3 Hz) was measured across the item presentation interval ($<TIME_WIN_START>$ to $<TIME_WIN_END>$ ms). At each electrode, a t-test compared low theta for subsequently recalled vs subsequently forgotten items. \textbf{Surface Electrodes:} Red - significant positive effect (subsequently \textit{remembered} $>$ subsequently \textit{forgotten}). Blue - significant negative effect (subsequently \textit{remembered} $<$ subsequently \textit{forgotten}). Black - difference not significant. \textbf{Depth Electrodes:} All bipolar pairs shown in descending order of significance Bipolar pairs that exceed significance threshold ($p < .05$) are in boldface.}
\end{longtable}
\end{center}

\clearpage

\begin{center} 
\textbf{\large Significant Electrodes High Theta} \\
\begin{longtable}{C{.75cm} C{2cm} C{2.5cm} C{5.5cm} C{1.25cm} C{1.25cm}}
\hline
Type & Channel \# & Electrode Pair & Atlas Loc & \textit{p} & \textit{t}-stat \\
<SIGNIFICANT_ELECTRODES_HTA>
\hline
\caption{Subsequent memory effect high theta. High theta activity (3-9 Hz) was measured across the item presentation interval ($<TIME_WIN_START>$ to $<TIME_WIN_END>$ ms). At each electrode, a t-test compared high theta for subsequently recalled vs subsequently forgotten items. \textbf{Surface Electrodes:} Red - significant positive effect (subsequently \textit{remembered}  $>$ subsequently \textit{forgotten}). Blue - significant negative effect (subsequently \textit{remembered} $<$ subsequently \textit{forgotten}). Black - difference not significant. \textbf{Depth Electrodes:} All bipolar pairs shown in descending order of significance Bipolar pairs that exceed significance threshold ($p < .05$) are in boldface.}
\end{longtable}
\end{center}

\clearpage

\begin{center} 
\textbf{\large Significant Electrodes Gamma} \\
\begin{longtable}{C{.75cm} C{2cm} C{2.5cm} C{5.5cm} C{1.25cm} C{1.25cm}}
\hline
Type & Channel \# & Electrode Pair & Atlas Loc & \textit{p} & \textit{t}-stat \\
<SIGNIFICANT_ELECTRODES_G>
\hline
\caption{Subsequent memory effect gamma. Gamma activity (40-70 Hz) was measured across the item presentation interval ($<TIME_WIN_START>$ to $<TIME_WIN_END>$ ms). At each electrode, a t-test compared gamma for subsequently recalled vs subsequently forgotten items. \textbf{Surface Electrodes:} Red - significant positive effect (subsequently \textit{remembered}  $>$ subsequently \textit{forgotten}). Blue - significant negative effect (subsequently \textit{remembered} $<$ subsequently \textit{forgotten}). Black - difference not significant. \textbf{Depth Electrodes:} All bipolar pairs shown in descending order of significance Bipolar pairs that exceed significance threshold ($p < .05$) are in boldface.}
\end{longtable}
\end{center}

\clearpage

\begin{center} 
\textbf{\large Significant Electrodes HFA} \\
\begin{longtable}{C{.75cm} C{2cm} C{2.5cm} C{5.5cm} C{1.25cm} C{1.25cm}}
\hline
Type & Channel \# & Electrode Pair & Atlas Loc & \textit{p} & \textit{t}-stat \\
<SIGNIFICANT_ELECTRODES_HFA>
\hline
\caption{Subsequent memory effect HFA. High frequency activity (HFA, 70-200 Hz) was measured across the item presentation interval ($<TIME_WIN_START>$ to $<TIME_WIN_END>$ ms). At each electrode, a t-test compared HFA for subsequently recalled vs subsequently forgotten items. \textbf{Surface Electrodes:} Red - significant positive effect (subsequently \textit{remembered} HFA $>$ subsequently \textit{forgotten} HFA). Blue - significant negative effect (subsequently \textit{remembered} HFA $<$ subsequently \textit{forgotten} HFA). Black - difference not significant. \textbf{Depth Electrodes:} All bipolar pairs shown in descending order of significance Bipolar pairs that exceed significance threshold ($p < .05$) are in boldface.}
\end{longtable}
\end{center}

\clearpage
\begin{center}
\textbf{\Large Multivariate classification analysis}
\end{center}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.45]{<ROC_AND_TERC_PLOT_FILE>}
\caption{\textbf{Left:} ROC curve for the subject for predicting \textit{recalled vs not recalled}; 
\textbf{Right:} Subject recall performance represented as
percentage deviation from the (subject) mean, separated by tercile
of the classifier encoding efficiency estimate for each encoded item.}
\end{figure}

$\bullet$ Area Under Curve = $<AUC>$\%

$\bullet$ Permutation test $p$-value: $<PERM-P-VALUE>$

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
