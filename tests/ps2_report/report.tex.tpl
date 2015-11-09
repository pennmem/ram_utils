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
\lhead{<HEADER_LEFT>}
\rhead{Date created: <DATE>}
\begin{document}


\section*{<SECTION_TITLE>}


<PT>

<PATIENT_TABLE>

\begin{tabular}{ccc}
\begin{minipage}[htbp]{250pt}
 \input{../RAM_FR_blurb.tex}
\begin{itemize}\item\textbf{Number of sessions: }1\item\textbf{Number of electrodes: }80\end{itemize}\end{minipage}
&
\begin{tabular}{|c|c|c|}
\hline Session \# & Date & Length (min)\\
\hline 0 & 27-Aug-2015 & 55.8 \\ 
\hline 
 \end{tabular} 
\end{tabular}
\vspace{0.3 in}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c|c}
\multicolumn{4}{c}{\textbf{Free Recall}} \\ 
\hline
300 words  &56 correct (18.7\%) &3 PLI (1.0\%) &2 ELI (0.7\%) \\ \hline 
\end{tabular}
\caption{An intrusion was a word that was vocalized during the retrieval period that was not studied on the most recent list. Intrusions were either words from a previous list (\textbf{PLI}: prior-list intrusions) or words that were not studied at all (\textbf{ELI}: extra-list intrusions).}
\end{table}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c}
\multicolumn{3}{c}{\textbf{Math distractor}} \\ 
\hline
76 math problems  &63 correct (82.9\%) &2.9 problems per list  \\ \hline 
\end{tabular}
\caption{After each list, the patient was given 20 seconds to perform as many arithmetic problems as possible, which served as a distractor before the beginning of recall.}
\end{table}

\begin{figure}[!h]
\centering
\subfigure[]{\includegraphics[scale=0.4]{~/reports/RAM_FR1/R1074M/analyses/behav/0/SPC_RAM_FR1_R1074M.eps}}
\subfigure[]{\includegraphics[scale=0.4]{~/reports/RAM_FR1/R1074M/analyses/behav/0/PFR_RAM_FR1_R1074M.eps}}
\caption{\textbf{Free recall:} (a) Overall probability of recall as a function of serial position. (b) Probability of FIRST recall as a function of serial position.}

\end{figure}

\clearpage

\begin{center}
\textbf{\large Significant Electrodes} \\ 
\begin{longtable}{C{.75cm} C{2.5cm} C{4cm} C{4cm}  C{1.25cm}  C{1.25cm}  }
\hline 
Type &Electrode Pair & Atlas Loc1 & Atlas Loc2  & \textit{p} & \textit{t}-stat \\ 
\hline 
 \textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LY9-LY10}}   & \textbf{\textcolor{BrickRed}{Left AnG angular gyrus}} & \textbf{\textcolor{BrickRed}{Left SMG supramarginal gyrus}}  & \textbf{\textcolor{BrickRed}{$<$.001}}  & \textbf{\textcolor{BrickRed}{3.686}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LW10-LW11}}   & \textbf{\textcolor{BrickRed}{Left MFG middle frontal gyrus}} & \textbf{\textcolor{BrickRed}{Left MFG middle frontal gyrus}}  & \textbf{\textcolor{BrickRed}{$<$.001}}  & \textbf{\textcolor{BrickRed}{3.349}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LX6-LX7}}   & \textbf{\textcolor{BrickRed}{Left Cerebral White Matter}} & \textbf{\textcolor{BrickRed}{Left Cerebral White Matter}}  & \textbf{\textcolor{BrickRed}{0.002}}  & \textbf{\textcolor{BrickRed}{3.055}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LX12-LX13}}   & \textbf{\textcolor{BrickRed}{Left MFG middle frontal gyrus}} & \textbf{\textcolor{BrickRed}{Left MFG middle frontal gyrus}}  & \textbf{\textcolor{BrickRed}{0.005}}  & \textbf{\textcolor{BrickRed}{2.83}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LY11-LY12}}   & \textbf{\textcolor{BrickRed}{Left Cerebral White Matter}} & \textbf{\textcolor{BrickRed}{Left SMG supramarginal gyrus}}  & \textbf{\textcolor{BrickRed}{0.01}}  & \textbf{\textcolor{BrickRed}{2.604}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LW8-LW9}}   & \textbf{\textcolor{BrickRed}{Left Cerebral White Matter}} & \textbf{\textcolor{BrickRed}{Left MFG middle frontal gyrus}}  & \textbf{\textcolor{BrickRed}{0.02}}  & \textbf{\textcolor{BrickRed}{2.339}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LY7-LY8}}   & \textbf{\textcolor{BrickRed}{Left SPL superior parietal lobule}} & \textbf{\textcolor{BrickRed}{Left SMG supramarginal gyrus}}  & \textbf{\textcolor{BrickRed}{0.022}}  & \textbf{\textcolor{BrickRed}{2.311}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LY8-LY9}}   & \textbf{\textcolor{BrickRed}{Left SMG supramarginal gyrus}} & \textbf{\textcolor{BrickRed}{Left AnG angular gyrus}}  & \textbf{\textcolor{BrickRed}{0.03}}  & \textbf{\textcolor{BrickRed}{2.175}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LFONE1-LFONE2}}   & \textbf{\textcolor{BrickRed}{Left MOrG medial orbital gyrus}} & \textbf{\textcolor{BrickRed}{Left MOrG medial orbital gyrus}}  & \textbf{\textcolor{BrickRed}{0.036}}  & \textbf{\textcolor{BrickRed}{2.11}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LY12-LY13}}   & \textbf{\textcolor{BrickRed}{Left SMG supramarginal gyrus}} & \textbf{\textcolor{BrickRed}{Left Cerebral White Matter}}  & \textbf{\textcolor{BrickRed}{0.039}}  & \textbf{\textcolor{BrickRed}{2.069}}  \\ 
 \hline 
D & LFONE12-LFONE13   & Left Cerebral White Matter & Left MFG middle frontal gyrus  & 0.074  & -1.791  \\ 
 \hline 
D & LX3-LX4   & Left Cerebral White Matter & Left Cerebral White Matter  & 0.076  & -1.782  \\ 
 \hline 
D & LN6-LN7   & Left Cerebral White Matter & Left SFG superior frontal gyrus  & 0.084  & 1.734  \\ 
 \hline 
D & LN4-LN5   & Left SFG superior frontal gyrus & Left SFG superior frontal gyrus  & 0.087  & 1.717  \\ 
 \hline 
D & LFONE3-LFONE4   & Left MOrG medial orbital gyrus & Left MOrG medial orbital gyrus  & 0.098  & -1.658  \\ 
 \hline 
D & LY10-LY11   & Left SMG supramarginal gyrus & Left Cerebral White Matter  & 0.104  & 1.63  \\ 
 \hline 
D & LY6-LY7   & Left Cerebral White Matter & Left SPL superior parietal lobule  & 0.125  & 1.539  \\ 
 \hline 

\hline 
\caption{Subsequent memory effect HFA FFT. High frequency activity (HFA, 70-200 Hz) was measured across the word presentation interval (0 to 1600ms). At each electrode, a t-test compared HFA for subsequently recalled vs subsequently forgotten items. \textbf{Surface Electrodes:} Red - significant positive effect (subsequently \textit{remembered} HFA $>$ subsequently \textit{forgotten} HFA). Blue - significant negative effect (subsequently \textit{remembered} HFA $<$ subsequently \textit{forgotten} HFA). Black - difference not significant. \textbf{Depth Electrodes:} All bipolar pairs shown in descending order of significance Bipolar pairs that exceed significance threshold ($p < .05$) are in boldface.}
\end{longtable}
\end{center}

\clearpage\begin{center}
\textbf{\large Significant Electrodes} \\ 
\begin{longtable}{C{.75cm} C{2.5cm} C{4cm} C{4cm}  C{1.25cm}  C{1.25cm}  }
\hline 
Type &Electrode Pair & Atlas Loc1 & Atlas Loc2  & \textit{p} & \textit{t}-stat \\ 
\hline 
 \textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LY8-LY9}}   & \textbf{\textcolor{BrickRed}{Left SMG supramarginal gyrus}} & \textbf{\textcolor{BrickRed}{Left AnG angular gyrus}}  & \textbf{\textcolor{BrickRed}{$<$.001}}  & \textbf{\textcolor{BrickRed}{4.441}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LY7-LY8}}   & \textbf{\textcolor{BrickRed}{Left SPL superior parietal lobule}} & \textbf{\textcolor{BrickRed}{Left SMG supramarginal gyrus}}  & \textbf{\textcolor{BrickRed}{$<$.001}}  & \textbf{\textcolor{BrickRed}{4.172}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LY10-LY11}}   & \textbf{\textcolor{BrickRed}{Left SMG supramarginal gyrus}} & \textbf{\textcolor{BrickRed}{Left Cerebral White Matter}}  & \textbf{\textcolor{BrickRed}{$<$.001}}  & \textbf{\textcolor{BrickRed}{3.944}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LW8-LW9}}   & \textbf{\textcolor{BrickRed}{Left Cerebral White Matter}} & \textbf{\textcolor{BrickRed}{Left MFG middle frontal gyrus}}  & \textbf{\textcolor{BrickRed}{$<$.001}}  & \textbf{\textcolor{BrickRed}{3.778}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LY11-LY12}}   & \textbf{\textcolor{BrickRed}{Left Cerebral White Matter}} & \textbf{\textcolor{BrickRed}{Left SMG supramarginal gyrus}}  & \textbf{\textcolor{BrickRed}{$<$.001}}  & \textbf{\textcolor{BrickRed}{3.613}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LW10-LW11}}   & \textbf{\textcolor{BrickRed}{Left MFG middle frontal gyrus}} & \textbf{\textcolor{BrickRed}{Left MFG middle frontal gyrus}}  & \textbf{\textcolor{BrickRed}{0.004}}  & \textbf{\textcolor{BrickRed}{2.893}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LX12-LX13}}   & \textbf{\textcolor{BrickRed}{Left MFG middle frontal gyrus}} & \textbf{\textcolor{BrickRed}{Left MFG middle frontal gyrus}}  & \textbf{\textcolor{BrickRed}{0.006}}  & \textbf{\textcolor{BrickRed}{2.768}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LY9-LY10}}   & \textbf{\textcolor{BrickRed}{Left AnG angular gyrus}} & \textbf{\textcolor{BrickRed}{Left SMG supramarginal gyrus}}  & \textbf{\textcolor{BrickRed}{0.007}}  & \textbf{\textcolor{BrickRed}{2.724}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LW7-LW8}}   & \textbf{\textcolor{BrickRed}{Left Cerebral White Matter}} & \textbf{\textcolor{BrickRed}{Left Cerebral White Matter}}  & \textbf{\textcolor{BrickRed}{0.008}}  & \textbf{\textcolor{BrickRed}{2.652}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LY6-LY7}}   & \textbf{\textcolor{BrickRed}{Left Cerebral White Matter}} & \textbf{\textcolor{BrickRed}{Left SPL superior parietal lobule}}  & \textbf{\textcolor{BrickRed}{0.01}}  & \textbf{\textcolor{BrickRed}{2.585}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LFONE1-LFONE2}}   & \textbf{\textcolor{BrickRed}{Left MOrG medial orbital gyrus}} & \textbf{\textcolor{BrickRed}{Left MOrG medial orbital gyrus}}  & \textbf{\textcolor{BrickRed}{0.011}}  & \textbf{\textcolor{BrickRed}{2.554}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LY1-LY2}}   & \textbf{\textcolor{BrickRed}{Left PCu precuneus}} & \textbf{\textcolor{BrickRed}{Left PCu precuneus}}  & \textbf{\textcolor{BrickRed}{0.012}}  & \textbf{\textcolor{BrickRed}{2.542}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LW9-LW10}}   & \textbf{\textcolor{BrickRed}{Left MFG middle frontal gyrus}} & \textbf{\textcolor{BrickRed}{Left MFG middle frontal gyrus}}  & \textbf{\textcolor{BrickRed}{0.028}}  & \textbf{\textcolor{BrickRed}{2.209}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LX8-LX9}}   & \textbf{\textcolor{BrickRed}{Left Cerebral White Matter}} & \textbf{\textcolor{BrickRed}{Left Cerebral White Matter}}  & \textbf{\textcolor{BrickRed}{0.03}}  & \textbf{\textcolor{BrickRed}{2.178}}  \\ 
 \hline 
\textbf{\textcolor{BrickRed}{D}} & \textbf{\textcolor{BrickRed}{LN7-LN8}}   & \textbf{\textcolor{BrickRed}{Left SFG superior frontal gyrus}} & \textbf{\textcolor{BrickRed}{}}  & \textbf{\textcolor{BrickRed}{0.038}}  & \textbf{\textcolor{BrickRed}{2.08}}  \\ 
 \hline 
D & LY4-LY5   & Left Cerebral White Matter & Left Cerebral White Matter  & 0.058  & -1.899  \\ 
 \hline 
D & LW3-LW4   & Left Cerebral White Matter & Left Cerebral White Matter  & 0.096  & 1.67  \\ 
 \hline 
D & LW2-LW3   & Left Cerebral White Matter & Left Cerebral White Matter  & 0.113  & 1.589  \\ 
 \hline 
D & LFONE13-LFONE14   & Left MFG middle frontal gyrus & Left MFG middle frontal gyrus  & 0.117  & 1.572  \\ 
 \hline 
D & LX4-LX5   & Left Cerebral White Matter & Left Cerebral White Matter  & 0.124  & -1.541  \\ 
 \hline 
D & LW6-LW7   & Left Cerebral White Matter & Left Cerebral White Matter  & 0.134  & 1.504  \\ 
 \hline 

\hline 
\caption{Subsequent memory effect HFA. High frequency activity (HFA, 70-200 Hz) was measured across the word presentation interval (0 to 1600ms). At each electrode, a t-test compared HFA for subsequently recalled vs subsequently forgotten items. \textbf{Surface Electrodes:} Red - significant positive effect (subsequently \textit{remembered} HFA $>$ subsequently \textit{forgotten} HFA). Blue - significant negative effect (subsequently \textit{remembered} HFA $<$ subsequently \textit{forgotten} HFA). Black - difference not significant. \textbf{Depth Electrodes:} All bipolar pairs shown in descending order of significance Bipolar pairs that exceed significance threshold ($p < .05$) are in boldface.}
\end{longtable}
\end{center}

\clearpage\begin{center}
\textbf{\large Significant Electrodes} \\ 
\begin{longtable}{C{.75cm} C{2.5cm} C{4cm} C{4cm}  C{1.25cm}  C{1.25cm}  }
\hline 
Type &Electrode Pair & Atlas Loc1 & Atlas Loc2  & \textit{p} & \textit{t}-stat \\ 
\hline 
 \multicolumn{5}{c}{No significant electrodes} \\ 
 \hline 

\hline 
\caption{Correct recall effect HFA. High frequency activity (HFA, 70-200 Hz) was measured during the recall test. At each electrode, a t-test compared HFA for the recall (-750 to -50 ms pre-vocalization) and deliberation (-1500 to -800 ms pre-vocalization) periods. \textbf{Surface Electrodes:} Bipolar pairs exceeding significance threshold ($p < .05$) for significance. Red - significant positive effect (\textit{recall} HFA $>$ \textit{deliberation} HFA). Blue - significant negative effect (\textit{recall} HFA $<$ \textit{deliberation} HFA). Black - difference not significant. \textbf{Depth Electrodes:} All bipolar pairs shown in descending order of significance (uncorrected). Bipolar pairs that exceed the FDR threshold are in boldface. }
\end{longtable}
\end{center}

\clearpage\end{document}


