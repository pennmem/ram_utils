\documentclass[a4paper]{article} 
\usepackage[usenames,dvipsnames,svgnames,table]{xcolor}
\usepackage{graphicx,multirow} 
\usepackage{longtable}
\usepackage{pdfpages}
\usepackage{mathtools}
\usepackage{array}
\usepackage{enumitem}
\usepackage{booktabs}
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
\lhead{RAM FR1 \& CatFR1 connectivity report v 1.0}
\chead{Subject: \textbf{<SUBJECT>}}
\rhead{Date created: <DATE>}
\begin{document}


\section*{<SUBJECT> RAM FR1 \& CatFR1 Connectivity Strength Report}

\begin{center}
\textbf{\large Significant Electrodes} \\
\begin{longtable}{C{.75cm} C{2cm} C{4cm} C{5.5cm} C{1.25cm}}
\hline
Type & Channel \# & Electrode Pair & Atlas Loc & $z$-score \\
<CONNECTIVITY_STRENGTH_TABLE>
\hline
\caption{Connectivity Strength Table (gamma, 45-95 Hz) All bipolar pairs shown in descending order of significance.}
\end{longtable}
\end{center}

\clearpage

\section*{APPENDIX: Connectivity Strength Calculation}

\begin{enumerate}
\item Compute Morlet wavelets for $[-1.0,2.8]$ sec interval around each word encoding. This report uses $11$
linspaced frequencies from $45$ to $95$ Hz.
\item For each bipolar pair (without a common electrode) and each frequency, compute phase differences of the
wavelets and average them across $19$ time bins, $200$ ms length each.
\item For each bipolar pair, frequency, and time bin, compute the resultant vector strength for the recalled
and non-recalled class and then compute the $f$-statistic using the formula:
\[ F_{pq}(f,t) = \frac{n_2(n_1-1)(1-\bar{R}_2)}{n_1(n_2-1)(1-\bar{R}_1)}, \]
where $(pq)$ is a pair of bipolar pairs without a common electrode, $f$ is a frequency, and $t$ is a time bin.
\item Generate $500$ permutations of recalls labels and compute the $f$-statistics for them (null distribution).
\item For each $f$-statistic (either true or permuted), assess its $z$-score against the generated null
distribution.
\item Average all $z$-scores across frequencies and time bins.
\item The connectivity strength is negatives of $z$-scores of true $f$-statistic $z$-scores against $z$-scores
of permuted $f$-statistics.

\end{enumerate}

\end{document}
