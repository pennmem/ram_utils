\documentclass[a4paper]{article}

\usepackage[table]{xcolor}
\usepackage{graphicx}
\usepackage{grffile}
\usepackage{caption}
\usepackage{booktabs}
\usepackage[skip=0pt]{subcaption}
\usepackage{morefloats}
\errorcontextlines 1000

\addtolength{\oddsidemargin}{-.875in}
\addtolength{\evensidemargin}{-.875in}
\addtolength{\textwidth}{1.75in}
\addtolength{\topmargin}{-.75in}
\addtolength{\textheight}{1.75in}

\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\lhead{ RAM PS4/<TASK> report v 1.0}
\chead{Subject: \textbf{ <SUBJECT> }}
\rhead{Date created: <DATE>}
\begin{document}

\section*{ <SUBJECT> RAM PS4 Parameter Search Report}

\begin{minipage}{0.5\textwidth}
\begin{itemize}
  \item \textbf{Number of sessions:} <NUMBER_OF_PS4_SESSIONS>
\end{itemize}
\end{minipage}
\begin{minipage}{0.5\textwidth}
\begin{tabular}{|l|c|c|c|}
\hline Session \# & Date & Length (min) & \# Lists \\
<SESSION_DATA>
\end{tabular}
\end{minipage}

<PS4_SESSION_PAGES>

\subsection*{Post-stim EEG}
\begin{figure}[ht!]
\includegraphics[scale=0.7]{<POST_STIM_EEG>}
\caption{Post-stimulation EEG signal,averaged across events}
\end{figure}
\end{document}


