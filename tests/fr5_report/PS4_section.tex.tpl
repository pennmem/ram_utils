{% block  PS4_SECTION %}
\section*{ {{SUBJECT}} RAM PS4 Parameter Search Report}

\begin{minipage}{0.5\textwidth}
\begin{itemize}
  \item \textbf{Number of sessions:} {{NUMBER_OF_PS4_SESSIONS}}
  \item \textbf{Number of electrodes:} {{NUMBER_OF_ELECTRODES}}
\end{itemize}
\end{minipage}
\begin{minipage}{0.5\textwidth}
\begin{tabular}{|c|c|c|}
\hline Session \# & Date & Length (min) \\
{{SESSION_DATA}}
\end{tabular}

\end{minipage}
{% for sess in PS4_SESSION_SUMMARIES%}
    {% if sess != -1 %}
\clearpage

\section*{Session {{sess}} }

\begin{figure}[ht!]
\renewcommand{\thesubfigure}{i}
\begin{subfigure}[!h]{\linewidth}
\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{ {{SESSION_SUMMARIES[sess].LOC_1_PLOT_FILE}} }
\subcaption{{{SESSION_SUMMARIES[sess].LOC_1}}}
\end{subfigure}
\renewcommand{\thesubfigure}{ii}
\begin{subfigure}[!h]{\linewidth}
\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{ {{SESSION_SUMMARIES[sess].LOC_2_PLOT_FILE}} }
\subcaption{{{SESSION_SUMMARIES[sess].LOC_2}}}
\end{subfigure}

\caption{Classifier response as a function of amplitude}
\end{figure}
    {% end %}
{% end %}
{% if PREFERRED_LOCATION %}
\begin{itemize}
 \item \textbf{ Preferred location: {{PREFERRED_LOCATION}}}
 \item \textbf{ Preferred amplitude: {{PREFERRED_AMPLITUDE}} }
 \item \textbf{ t-stat = {{TSTAT}} }
 \item \textbf{ {{PVALUE}} }
\end{itemize}
{% endblock %}




