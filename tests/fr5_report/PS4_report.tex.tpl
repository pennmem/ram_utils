\documentclass[a4paper]{article}

\usepackage[table]{xcolor}
\usepackage{graphicx}
\usepackage{grffile}
\usepackage{caption}
\usepackage[skip=0pt]{subcaption}
\usepackage{morefloats}

\addtolength{\oddsidemargin}{-.875in}
\addtolength{\evensidemargin}{-.875in}
\addtolength{\textwidth}{1.75in}
\addtolength{\topmargin}{-.75in}
\addtolength{\textheight}{1.75in}

\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\lhead{ {{EXPERIMENT}} report v 1.0}
\chead{Subject: \textbf{ {{SUBJECT}} !}}
\rhead{Date created: {DATE}}
\begin{document}

\section*{ {{SUBJECT}} RAM {{EXPERIMENT}} Parameter Search Report}

\begin{minipage}{0.5\textwidth}
\begin{itemize}
  \item \textbf{Number of sessions:} {{NUMBER_OF_SESSIONS}}
  \item \textbf{Number of electrodes:} {{NUMBER_OF_ELECTRODES}}
\end{itemize}
\end{minipage}
\begin{minipage}{0.5\textwidth}
\begin{tabular}{|c|c|c|}
\hline Session \# & Date & Length (min) \\
{{SESSION_DATA}}
\end{tabular}

\end{minipage}
{% for sess in SESSION_SUMMARIES:%}
\clearpage

\section*{Session {{sess}} }

\begin{figure}[ht!]
\renewcommand{\thesubfigure}{i}
\begin{subfigure}[!h]{\linewidth}
\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{ {{SESSION_SUMMARIES[sess][LOC_1_PLOT_FILE]}} }
\subcaption{{{SESSION_SUMMARIES[sess][LOC_1]}}}
\end{subfigure}
\renewcommand{\thesubfigure}{ii}
\begin{subfigure}[!h]{\linewidth}
\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{ {{SESSION_SUMMARIES[sess][LOC_2_PLOT_FILE]}} }
\subcaption{{{SESSION_SUMMARIES[sess][LOC_2]}}}
\end{subfigure}

\caption{Classifier response as a function of amplitude}
\end{figure}



