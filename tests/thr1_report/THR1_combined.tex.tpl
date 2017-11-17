\documentclass[a4paper]{article}
\usepackage[usenames,dvipsnames,svgnames,table]{xcolor}
\usepackage{graphicx,multirow}
%\usepackage{epstopdf}
%\usepackage{wrapfig}
\usepackage{longtable}
\usepackage{pdfpages}
%\usepackage{mathtools}
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
\lhead{RAM THR1 report v 2.12}
\chead{Subject: \textbf{<SUBJECT>}}
\rhead{Date created: <DATE>}
\begin{document}


\section*{<SUBJECT> RAM THR1 Report}

\begin{tabular}{ccc}
\begin{minipage}[htbp]{200pt}
In the Treasure Hunt Recall (THR), participants navigate a virtual environment and encounter objects hidden in treasure chests. Participants are then probed with a location and verbally recall the associated item.
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
\begin{tabular}{c|c}
\multicolumn{2}{c}{\textbf{Performance SessionSummary}} \\
\hline
$<N_WORDS>$ words & $<N_CORRECT_WORDS>$ correct ($<PC_CORRECT_WORDS>$\%)  \\ \hline
\end{tabular}
\caption{Correct items are items that were accurately identified within the 6-second probe period.}
\end{table}

\begin{figure}[!h]
\centering
\includegraphics[scale=0.4]{<PROB_RECALL_PLOT_FILE>}
\caption{(a) Overall probability of recall as a function of serial position. (b) Overall probability of recall as a function of probe position.}
\end{figure}

\clearpage

\begin{center}
\textbf{\large Significant Electrodes Low Theta} \\
\begin{longtable}{C{.75cm} C{2cm} C{2.5cm} C{5.5cm} C{1.25cm} C{1.25cm}}
\hline
Type & Channel \# & Electrode Pair & Atlas Loc & \textit{p} & \textit{t}-stat \\
<SIGNIFICANT_ELECTRODES_LOW_THETA>
\hline
\caption{Subsequent memory effect Low Theta. Low theta activity (1-3 Hz) was measured across the word presentation interval (0 to 1500 ms). At each electrode, a t-test compared activity for subsequently recalled vs subsequently forgotten items. \textbf{Surface Electrodes:} Red - significant positive effect (subsequently \textit{remembered} power $>$ subsequently \textit{forgotten} power). Blue - significant negative effect (subsequently \textit{remembered} power $<$ subsequently \textit{forgotten} power). Black - difference not significant. \textbf{Depth Electrodes:} All bipolar pairs shown in descending order of significance Bipolar pairs that exceed significance threshold ($p < .05$) are in boldface.}
\end{longtable}
\end{center}

\clearpage

\begin{center}
\textbf{\large Significant Electrodes High Theta} \\
\begin{longtable}{C{.75cm} C{2cm} C{2.5cm} C{5.5cm} C{1.25cm} C{1.25cm}}
\hline
Type & Channel \# & Electrode Pair & Atlas Loc & \textit{p} & \textit{t}-stat \\
<SIGNIFICANT_ELECTRODES_HIGH_THETA>
\hline
\caption{Subsequent memory effect High Theta. High theta activity (3-9 Hz) was measured across the word presentation interval (0 to 1500 ms). At each electrode, a t-test compared activity for subsequently recalled vs subsequently forgotten items. \textbf{Surface Electrodes:} Red - significant positive effect (subsequently \textit{remembered} power $>$ subsequently \textit{forgotten} power). Blue - significant negative effect (subsequently \textit{remembered} power $<$ subsequently \textit{forgotten} power). Black - difference not significant. \textbf{Depth Electrodes:} All bipolar pairs shown in descending order of significance Bipolar pairs that exceed significance threshold ($p < .05$) are in boldface.}
\end{longtable}
\end{center}

\clearpage

\begin{center}
\textbf{\large Significant Electrodes Gamma} \\
\begin{longtable}{C{.75cm} C{2cm} C{2.5cm} C{5.5cm} C{1.25cm} C{1.25cm}}
\hline
Type & Channel \# & Electrode Pair & Atlas Loc & \textit{p} & \textit{t}-stat \\
<SIGNIFICANT_ELECTRODES_GAMMA>
\hline
\caption{Subsequent memory effect Gamma. Gamma (30-70 Hz) was measured across the word presentation interval (0 to 1500 ms). At each electrode, a t-test compared activity for subsequently recalled vs subsequently forgotten items. \textbf{Surface Electrodes:} Red - significant positive effect (subsequently \textit{remembered} power $>$ subsequently \textit{forgotten} power). Blue - significant negative effect (subsequently \textit{remembered} power $<$ subsequently \textit{forgotten} power). Black - difference not significant. \textbf{Depth Electrodes:} All bipolar pairs shown in descending order of significance Bipolar pairs that exceed significance threshold ($p < .05$) are in boldface.}
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
\caption{Subsequent memory effect High Frequency Activity (HFA). HFA (70-200 Hz) was measured across the word presentation interval (0 to 1500 ms). At each electrode, a t-test compared activity for subsequently recalled vs subsequently forgotten items. \textbf{Surface Electrodes:} Red - significant positive effect (subsequently \textit{remembered} power $>$ subsequently \textit{forgotten} power). Blue - significant negative effect (subsequently \textit{remembered} power $<$ subsequently \textit{forgotten} power). Black - difference not significant. \textbf{Depth Electrodes:} All bipolar pairs shown in descending order of significance Bipolar pairs that exceed significance threshold ($p < .05$) are in boldface.}
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
