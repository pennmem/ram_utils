\documentclass[a4paper]{article} 

\usepackage[table]{xcolor}
\usepackage{graphicx}
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
\lhead{<EXPERIMENT> report v 3.1}
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

\begin{figure}[!h]
\centering
%\begin{subfigure}[!h]{\linewidth}
%\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{<CUMULATIVE_LOW_QUANTILE_PLOT_FILE>}
%\subcaption{Lower Half of Pre-Stim Classifier Output}
%\end{subfigure}
%\begin{subfigure}[!h]{\linewidth}
%\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{<CUMULATIVE_HIGH_QUANTILE_PLOT_FILE>}
%\subcaption{Upper Half of Pre-Stim Classifier Output}
%\end{subfigure}
%\begin{subfigure}[!h]{\linewidth}
\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{<CUMULATIVE_ALL_PLOT_FILE>}
\caption{\textbf{(a)} Means and standard errors of difference in classifier output post- and pre-stim;
\textbf{(b)} Means and standard errors of}
\[ \textrm{Expected Recall Change} = \frac{N_{recalls}(C \leq C_{\textrm{post}})/N_{items}(C \leq C_{\textrm{post}}) - N_{recalls}(C \leq C_{\textrm{pre}})/N_{items}(C \leq C_{\textrm{pre}})}{N_{recalls}/N_{items}}, \]
\raggedright
where $C_{\textrm{pre}}$ is pre-stim classifier output, $C_{\textrm{post}}$ is post-stim classifier output.
%\subcaption{All Trials}
%\end{subfigure}
\end{figure}

<CUMULATIVE_ADHOC_PAGE_TITLE>

<CUMULATIVE_PARAM1_TTEST_TABLE>
<CUMULATIVE_PARAM2_TTEST_TABLE>
<CUMULATIVE_PARAM12_TTEST_TABLE>

\end{document}
