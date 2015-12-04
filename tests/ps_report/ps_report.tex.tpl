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
\lhead{<EXPERIMENT> report v 1.0}
\chead{Subject: \textbf{<SUBJECT_ID>}}
\rhead{Date created: <DATE>}
\begin{document}

\section*{<SUBJECT_ID> RAM <EXPERIMENT> Parameter Search Report}

\begin{minipage}{0.5\textwidth}
\begin{itemize}
  \item \textbf{Number of sessions:} <NUMBER_OF_SESSIONS>
  \item \textbf{Number of electrodes:} <NUMBER_OF_ELECTRODES>
\end{itemize}
\end{minipage}
\begin{minipage}{0.5\textwidth}
<SESSION_DATA>

% \begin{tabular}{|c|c|c|}
% \hline Session \# & Date & Length (min) \\
% \hline 2 & 01-Oct-2015 & 26.7 \\
% \hline
% \end{tabular}

\end{minipage}

\clearpage

<REPORT_PAGES>

\textbf{Parameters:}
\begin{itemize}
  \item ISI: $<CUMULATIVE_ISI_MID>$ ($\pm <CUMULATIVE_ISI_HALF_RANGE>$) ms
  \item All channels
\end{itemize}

\begin{figure}[!h]
\centering
\subfigure{\includegraphics[trim={2.5cm 2.5cm 3.5cm 3.5cm},clip,scale=0.53]{<CUMULATIVE_PLOT_FILE>}}
\caption{Left: <CUMULATIVE_PARAMETER1> vs classifier delta. Right: <CUMULATIVE_PARAMETER2> vs classifier delta. All trials: (a,b). Lowest tercile: (c,d). Highest tercile: (e,f).}
\end{figure}

\end{document}
