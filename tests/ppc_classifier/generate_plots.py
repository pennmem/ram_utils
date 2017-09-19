import sys

import matplotlib

matplotlib.use('Agg')
#matplotlib.use('ps')
from matplotlib import rc

rc('text',usetex=True)
rc('text.latex', preamble='\usepackage{color}\n\usepackage[dvipsnames]{xcolor}')
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages

from sklearn.externals import joblib

sys.path.append('/home1/busygin/ram_utils/tests/fr1_report')


fig = plt.figure(1)
f, axarr = plt.subplots(2, 2)


subject = 'R1065J'

#plt.subplot(221)

ppc_fpr = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-ppc_only_fpr.pkl' % (subject,subject))
ppc_tpr = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-ppc_only_tpr.pkl' % (subject,subject))
ppc_auc = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-ppc_only_auc.pkl' % (subject,subject))

fpr = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-fpr.pkl' % (subject,subject))
tpr = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-tpr.pkl' % (subject,subject))
auc = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-auc.pkl' % (subject,subject))

pow_xval = joblib.load('/scratch/mswat/automated_reports/FR1_reports/RAM_FR1_%s/%s-RAM_FR1-xval_output.pkl' % (subject,subject))
pow_xval = pow_xval[-1]
pow_fpr = pow_xval.fpr
pow_tpr = pow_xval.tpr
pow_auc = pow_xval.auc

axarr[0, 0].set_title(r'{\small \textcolor{black}{%s} \textcolor{blue}{AUC=%.2f\%%} \textcolor{red}{AUC=%.2f\%%} \textcolor{OliveGreen}{AUC=%.2f\%%}}' % (subject,ppc_auc*100,auc*100,pow_auc*100))
axarr[0, 0].plot(ppc_fpr, ppc_tpr)
axarr[0, 0].plot(pow_fpr, pow_tpr)
axarr[0, 0].plot(fpr, tpr)


subject = 'R1135E'

#plt.subplot(222)

ppc_fpr = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-ppc_only_fpr.pkl' % (subject,subject))
ppc_tpr = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-ppc_only_tpr.pkl' % (subject,subject))
ppc_auc = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-ppc_only_auc.pkl' % (subject,subject))

fpr = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-fpr.pkl' % (subject,subject))
tpr = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-tpr.pkl' % (subject,subject))
auc = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-auc.pkl' % (subject,subject))

pow_xval = joblib.load('/scratch/mswat/automated_reports/FR1_reports/RAM_FR1_%s/%s-RAM_FR1-xval_output.pkl' % (subject,subject))
pow_xval = pow_xval[-1]
pow_fpr = pow_xval.fpr
pow_tpr = pow_xval.tpr
pow_auc = pow_xval.auc

axarr[0, 1].set_title(r'{\small \textcolor{black}{%s} \textcolor{blue}{AUC=%.2f\%%} \textcolor{red}{AUC=%.2f\%%} \textcolor{OliveGreen}{AUC=%.2f\%%}}' % (subject,ppc_auc*100,auc*100,pow_auc*100))
axarr[0, 1].plot(ppc_fpr, ppc_tpr)
axarr[0, 1].plot(pow_fpr, pow_tpr)
axarr[0, 1].plot(fpr, tpr)


subject = 'R1145J_1'

#plt.subplot(223)

ppc_fpr = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-ppc_only_fpr.pkl' % (subject,subject))
ppc_tpr = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-ppc_only_tpr.pkl' % (subject,subject))
ppc_auc = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-ppc_only_auc.pkl' % (subject,subject))

fpr = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-fpr.pkl' % (subject,subject))
tpr = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-tpr.pkl' % (subject,subject))
auc = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-auc.pkl' % (subject,subject))

pow_xval = joblib.load('/scratch/mswat/automated_reports/FR1_reports/RAM_FR1_%s/%s-RAM_FR1-xval_output.pkl' % (subject,subject))
pow_xval = pow_xval[-1]
pow_fpr = pow_xval.fpr
pow_tpr = pow_xval.tpr
pow_auc = pow_xval.auc

axarr[1, 0].set_title(r'{\small \textcolor{black}{%s} \textcolor{blue}{AUC=%.2f\%%} \textcolor{red}{AUC=%.2f\%%} \textcolor{OliveGreen}{AUC=%.2f\%%}}' % (subject.replace('_','\\textunderscore'),ppc_auc*100,auc*100,pow_auc*100))
axarr[1, 0].plot(ppc_fpr, ppc_tpr)
axarr[1, 0].plot(pow_fpr, pow_tpr)
axarr[1, 0].plot(fpr, tpr)

subject = 'R1156D'

#plt.subplot(224)

ppc_fpr = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-ppc_only_fpr.pkl' % (subject,subject))
ppc_tpr = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-ppc_only_tpr.pkl' % (subject,subject))
ppc_auc = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-ppc_only_auc.pkl' % (subject,subject))

fpr = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-fpr.pkl' % (subject,subject))
tpr = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-tpr.pkl' % (subject,subject))
auc = joblib.load('/scratch/busygin/FR1_ppc_same_class/%s/%s-RAM_FR1-auc.pkl' % (subject,subject))

pow_xval = joblib.load('/scratch/mswat/automated_reports/FR1_reports/RAM_FR1_%s/%s-RAM_FR1-xval_output.pkl' % (subject,subject))
pow_xval = pow_xval[-1]
pow_fpr = pow_xval.fpr
pow_tpr = pow_xval.tpr
pow_auc = pow_xval.auc

axarr[1, 1].set_title(r'{\small \textcolor{black}{%s} \textcolor{blue}{AUC=%.2f\%%} \textcolor{red}{AUC=%.2f\%%} \textcolor{OliveGreen}{AUC=%.2f\%%}}' % (subject,ppc_auc*100,auc*100,pow_auc*100))
axarr[1, 1].plot(ppc_fpr, ppc_tpr)
axarr[1, 1].plot(pow_fpr, pow_tpr)
axarr[1, 1].plot(fpr, tpr)

#fig.title('Comparison of FR1 classifiers')
plt.figtext(0.1,-0.1,r'\textcolor{blue}{Blue: PPC features only;} \textcolor{red}{Red: both PPC and power features;} \textcolor{OliveGreen}{Green: power features only.}')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.savefig('/scratch/busygin/ppc_plots.ps')

#with PdfPages('/scratch/busygin/ppc_plots.pdf') as pdf:
#    pdf.savefig()
#    plt.clf()
