\clearpage

\section*{Session <SESSION> }

\begin{figure}[ht!]
\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{ <SESSION_SUMMARIES[sess].PS_PLOT_FILE> }
\caption{Classifier response as a function of amplitude}
\end{figure}
    {% end %}
{% end %}
{% if PREFERRED_LOCATION %}
\begin{itemize}
 \item \textbf{ Preferred location: <PREFERRED_LOCATION>}
 \item \textbf{ Preferred amplitude: <PREFERRED_AMPLITUDE> }
 \item \textbf{ t-stat = <TSTAT> }
 \item \textbf{ <PVALUE> }
\end{itemize}
