
\clearpage

\begin{center}
\textbf{\Large <STIMTAG> (<REGION>), <FREQUENCY> Hz, Session(s): <SESSIONS>}
\end{center}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c}
\multicolumn{3}{c}{\textbf{Treasure Hunt}} \\
\hline
$<N_WORDS>$ words & $<N_CORRECT_WORDS>$ correct ($<PC_CORRECT_WORDS>$\%) & Generalizability AUC: $<AUC>$, p-val: $<AUC_P>$ \\ \hline
\end{tabular}
\caption{Items are considered to be correctly retrieved if confidence is not low AND if the distance error is below the median of the patient's TH1 responses. AUC is calculated by applying the TH1 classifier to all items in non-stim lists.}
\end{table}

% \begin{table}[!h]
% \centering
% \begin{tabular}{c|c|c}
% \multicolumn{3}{c}{\textbf{Math distractor}} \\
% \hline
% $<N_MATH>$ math problems & $<N_CORRECT_MATH>$ correct ($<PC_CORRECT_MATH>$\%) & $<MATH_PER_LIST>$ problems per list  \\ \hline
% \end{tabular}
% \caption{After each list, the patient was given 20 seconds to perform as many arithmetic problems as possible, which served as a distractor before the beginning of recall.}
% \end{table}

\begin{figure}[!h]
\centering
\subfigure{\includegraphics[scale=0.4]{<PROB_RECALL_PLOT_FILE>}}
\caption{\textbf{Treasure Hunt:} (a) Overall probability of recall as a function of serial position. (b) Probability of distance error, plotted separately for stim and non-stim items.}
\end{figure}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c|c}
\multicolumn{4}{c}{\textbf{Stim vs Non-Stim Recalls (list level)}} \\
\hline
$<N_CORRECT_STIM>/<N_TOTAL_STIM>$ ($<PC_FROM_STIM>$\%) from stim lists & $<N_CORRECT_NONSTIM>/<N_TOTAL_NONSTIM>$ ($<PC_FROM_NONSTIM>$\%) from non-stim lists & $\chi^2(1)=<CHISQR>$ & $p=<PVALUE>$ \\
\hline
\end{tabular}
\end{table}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c|c}
\multicolumn{4}{c}{\textbf{Stim vs Non-Stim Recalls (item level)}} \\
\hline
$<N_CORRECT_STIM_ITEM>/<N_TOTAL_STIM_ITEM>$ ($<PC_FROM_STIM_ITEM>$\%) from stim items & $<N_CORRECT_NONSTIM_ITEM>/<N_TOTAL_NONSTIM_ITEM>$ ($<PC_FROM_NONSTIM_ITEM>$\%) from non-stim items & $\chi^2(1)=<CHISQR_ITEM>$ & $p=<PVALUE_ITEM>$ \\
\hline
\end{tabular}
\end{table}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c|c}
\multicolumn{4}{c}{\textbf{Post Stim vs Non-Stim Recalls (item level)}} \\
\hline
$<N_CORRECT_POST_STIM_ITEM>/<N_TOTAL_POST_STIM_ITEM>$ ($<PC_FROM_POST_STIM_ITEM>$\%) from stim items & $<N_CORRECT_POST_NONSTIM_ITEM>/<N_TOTAL_POST_NONSTIM_ITEM>$ ($<PC_FROM_POST_NONSTIM_ITEM>$\%) from non-stim items & $\chi^2(1)=<CHISQR_POST_ITEM>$ & $p=<PVALUE_POST_ITEM>$ \\
\hline
\end{tabular}
\end{table}

\begin{table}[!h]
\centering
\begin{tabular}{c|c|c|c}
\multicolumn{4}{c}{\textbf{Mid/High Stim vs Mid/High Non-Stim Confidence (item level)}} \\
\hline
$<N_CONF_STIM_ITEM>/<N_TOTAL_STIM_ITEM>$ ($<PC_CONF_STIM_ITEM>$\%) from stim items & $<N_CONF_NONSTIM_ITEM>/<N_TOTAL_NONSTIM_ITEM>$ ($<PC_CONF_NONSTIM_ITEM>$\%) from non-stim items & $\chi^2(1)=<CHISQR_CONF>$ & $p=<PVALUE_CONF>$ \\
\hline
\end{tabular}
\end{table}

\clearpage


\begin{center}
\textbf{\Large Stim and recall analysis}
\end{center}

\begin{figure}[!h]
\centering
\subfigure{\includegraphics[scale=0.4]{<STIM_AND_RECALL_PLOT_FILE>}}
\caption*{\textbf{Stim and recall:} Distance error as a function of item number in the session. Red dots indicate stimulated items, blue dots indicate non-stimulated items from stim lists, and black dots indicate non-stim lists. The bottom row indicates stim/non-stim list status. Dashed line indicates recall threshold.}
\end{figure}

<BIOMARKER_PLOTS>
