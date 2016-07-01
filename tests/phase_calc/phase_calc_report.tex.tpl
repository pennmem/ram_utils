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
\lhead{RAM FR1 connectivity report v 1.0}
\chead{Subject: \textbf{<SUBJECT>}}
\rhead{Date created: <DATE>}
\begin{document}


\section*{<SUBJECT> RAM FR1 Connectivity Strength Report}

\begin{center}
\textbf{\large Significant Electrodes} \\
\begin{longtable}{C{.75cm} C{4cm} C{8cm} C{1.25cm}}
\hline
Type & Electrode Pair & Atlas Loc & $z$-score \\
<CONNECTIVITY_STRENGTH_TABLE>
\hline
\caption{Connectivity Strength Table (gamma, 45-95 Hz) All bipolar pairs shown in descending order of significance.}
\end{longtable}
\end{center}

\clearpage

\end{document}
