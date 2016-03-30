
\clearpage

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
\textbf{(b)} Means and standard errors of {\em Expected Recall Change:}}
%\[ \textrm{Expected Recall Change} = \frac{N_{recalls}(C \leq C_{\textrm{post}})/N_{items}(C \leq C_{\textrm{post}}) - N_{recalls}(C \leq C_{\textrm{pre}})/N_{items}(C \leq C_{\textrm{pre}})}{N_{recalls}/N_{items}}, \]
%\raggedright
%where $C_{\textrm{pre}}$ is pre-stim classifier output, $C_{\textrm{post}}$ is post-stim classifier output.
%\subcaption{All Trials}
%\end{subfigure}
\end{figure}
\[ \textrm{Expected Recall Change} = \left. \left( \frac{N_1 \Phi_1(\xi_{\textrm{post}})}{N_1 \Phi_1(\xi_{\textrm{post}}) + N_0 \Phi_0(\xi_{\textrm{post}})} - \frac{N_1 \Phi_1(\xi_{\textrm{pre}})}{N_1 \Phi_1(\xi_{\textrm{pre}}) + N_0 \Phi_0(\xi_{\textrm{pre}})}\right) \middle/ (N_1/N) \right., \]
$\bullet$ $N_1$ is \#recalls, $N_0$ is \#non-recalls, $N=N_1+N_0$; \\
$\bullet$ $\xi_{\textrm{post}} = \ln \frac{C_{\textrm{post}}}{1-C_{\textrm{post}}}$ is inverse logit of classifier post-stim output $C_{\textrm{post}}$; \\
$\bullet$ $\xi_{\textrm{pre}} = \ln \frac{C_{\textrm{pre}}}{1-C_{\textrm{pre}}}$ is inverse logit of classifier pre-stim output $C_{\textrm{pre}}$; \\
$\bullet$ $\Phi_1,\Phi_0$ are cdfs of Normal fits for inverse logit of classifier output for recalls/non-recalls with equal variance.
%$\bullet$ $\Phi_0 = {\cal N}(\mu_0,\sigma)$ is Normal fit for inverse logit of classifier output for non-recalls.

\vspace{3pc}

<TTEST_AGAINST_ZERO_TABLE>
