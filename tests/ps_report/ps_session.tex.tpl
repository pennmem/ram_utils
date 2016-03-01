
\section*{\hfil Session <SESS_NUM> \hfil}

\begin{tabular}{ccc}
\begin{minipage}[htbp]{160pt}
\textbf{Parameters:} \\
$\bullet$ <CONSTANT_NAME>: $<CONSTANT_VALUE>$ <CONSTANT_UNIT> \\
$\bullet$ ISI: $<ISI_MID>$ ($\pm <ISI_HALF_RANGE>$) ms \\
$\bullet$ Channel: <STIMTAG> \\
$\bullet$ Region: <REGION>
\end{minipage}
&
\begin{minipage}[htbp]{280pt}
\centering
\textbf{Two-factor ANOVA}

\begin{tabular}{|c|c|c|c|}
\hline & <PARAMETER1> & <PARAMETER2> & <PARAMETER1> $\times$ <PARAMETER2> \\
\hline $F$ & $<FVALUE1>$ & $<FVALUE2>$ & $<FVALUE12>$ \\
\hline $p$ & $<PVALUE1>$ & $<PVALUE2>$ & $<PVALUE12>$ \\
\hline
\end{tabular}
\end{minipage}
\end{tabular}

\begin{figure}[!h]
%\begin{subfigure}[!h]{\linewidth}
%\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{<LOW_QUANTILE_PLOT_FILE>}
%\subcaption{Lower Half of Pre-Stim Classifier Output}
%\end{subfigure}
%\begin{subfigure}[!h]{\linewidth}
%\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{<HIGH_QUANTILE_PLOT_FILE>}
%\subcaption{Upper Half of Pre-Stim Classifier Output}
%\end{subfigure}
%\begin{subfigure}[!h]{\linewidth}
\includegraphics[trim={0.0cm 0.45cm 0.0cm 0.0cm},clip,scale=0.4]{<ALL_PLOT_FILE>}
\caption{\textbf{(a)} Means and standard errors of difference in classifier output post- and pre-stim;
\textbf{(b)} Means and standard errors of}
\[ \textrm{Expected Recall Change} = \frac{N_{recalls}(C \leq C_{\textrm{post}})/N_{items}(C \leq C_{\textrm{post}}) - N_{recalls}(C \leq C_{\textrm{pre}})/N_{items}(C \leq C_{\textrm{pre}})}{N_{recalls}/N_{items}}, \]
\raggedright
where $C_{\textrm{pre}}$ is pre-stim classifier output, $C_{\textrm{post}}$ is post-stim classifier output.
%\subcaption{All Trials}
%\end{subfigure}
\end{figure}

<ADHOC_PAGE_TITLE>

<PARAM1_TTEST_TABLE>
<PARAM2_TTEST_TABLE>
<PARAM12_TTEST_TABLE>

\clearpage
