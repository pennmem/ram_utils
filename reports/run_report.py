from ReportUtils import CMLParser
from fr1_report import fr1_report
from fr1_catfr1_joint_report import fr1_catfr1_joint_report
from fr5_report import fr5_report
from fr_catfr_joint_stim_report import fr_catfr_joint_stim_report
from fr_stim_report import fr_stim_report
from fr_connectivity_report import fr_connectivity_report
# from pal1_report import pal1_report
from pal1_sys3_report import pal1_report as pal1_sys3_report
from pal5_report import pal5_report
from pal_stim_report import pal_stim_report
from ps4_report import ps4_report
from th1_report import  th1_report
from th_stim_report import th_stim_report
from thr1_report import thr1_report



task_to_report_function = {
    'FR1':fr1_report,
    'catFR1':fr1_report,
    'FR1_catFR1_joint':fr1_catfr1_joint_report,
    'FR5':fr5_report,
    'FR3_catFR3_joint':fr_catfr_joint_stim_report,
    'FR3':fr_stim_report,
    'catFR3':fr_stim_report,
    'PAL1':pal1_sys3_report,
    'PAl3':pal_stim_report,
    'PAL5':pal5_report,
    'PS4':ps4_report,
    'TH1':th1_report,
    'THR1':thr1_report,
    'TH3':th_stim_report,
    'FR_connectivity':fr_connectivity_report
}

if __name__ == '__main__':
    args = CMLParser().parse()
    task_to_report_function[args.task].run_report(args)


