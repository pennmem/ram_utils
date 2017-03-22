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

{% if HAS_PS4 %}
{% include PS4_section.tex.tpl %}
{% end %}

{% if HAS_FR5 %}
{% include FR5_section.tex.tpl %}
{% end %}
\end{document}