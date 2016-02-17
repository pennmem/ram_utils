\documentclass[a4paper]{article} 
\usepackage[usenames,dvipsnames,svgnames,table]{xcolor}
\usepackage{graphicx,multirow} 
\usepackage{epstopdf} 
\usepackage{subfigure,amsmath} 
\usepackage{wrapfig}
\usepackage{booktabs}
\usepackage{longtable} 
\usepackage{pdfpages}
\usepackage{mathtools}
\usepackage{array}
\usepackage{enumitem}
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
\lhead{<EXPERIMENT> report v 2.3}
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
\subfigure{\includegraphics[scale=0.45]{<ROC_AND_TERC_PLOT_FILE>}}
\caption{\textbf{(a)} ROC curve for the subject;
\textbf{(b)} Subject recall performance represented as
percentage devation from the (subject) mean, separated by tercile
of the classifier encoding efficiency estimate for each encoded word.}
\end{figure}

$\bullet$ Area Under Curve = $<AUC>$\%

$\bullet$ Permutation test $p$-value $<PERM-P-VALUE>$

$\bullet$ Youden's $J$-stat threshold = $<J-THRESH>$

\clearpage

<REPORT_PAGES>

\section*{\hfil Combined Report \hfil}

\begin{tabular}{ccc}
\begin{minipage}[htbp]{160pt}
\textbf{Parameters:} \\
$\bullet$ ISI: $<CUMULATIVE_ISI_MID>$ ($\pm <CUMULATIVE_ISI_HALF_RANGE>$) ms \\
$\bullet$ All channels
\end{minipage}
&
\begin{minipage}[htbp]{280pt}
\centering
\textbf{Two-factor ANOVA}

\begin{tabular}{|c|c|c|c|}
\hline & <CUMULATIVE_PARAMETER1> & <CUMULATIVE_PARAMETER2> & <CUMULATIVE_PARAMETER1> $\times$ <CUMULATIVE_PARAMETER2> \\
\hline $F$ & $<CUMULATIVE_FVALUE1>$ & $<CUMULATIVE_FVALUE2>$ & $<CUMULATIVE_FVALUE12>$ \\
\hline $p$ & $<CUMULATIVE_PVALUE1>$ & $<CUMULATIVE_PVALUE2>$ & $<CUMULATIVE_PVALUE12>$ \\
\hline
\end{tabular}
\end{minipage}
\end{tabular}

%\textbf{Parameters:}
%\begin{itemize}
%  \item ISI: $<CUMULATIVE_ISI_MID>$ ($\pm <CUMULATIVE_ISI_HALF_RANGE>$) ms
%  \item All channels
%\end{itemize}

\begin{figure}[!h]
\centering
\subfigure{\includegraphics[scale=0.35]{<CUMULATIVE_PLOT_FILE>}}
\end{figure}

<CUMULATIVE_PARAM1_TTEST_TABLE>
<CUMULATIVE_PARAM2_TTEST_TABLE>
<CUMULATIVE_PARAM12_TTEST_TABLE>

\end{document}
