import json
import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from ramutils.utils import encode_file

def make_navtime_plot(evs):

    plot_buffer = io.BytesIO()
    
    word_evs = evs.query('type == "WORD"')
    total_delivery_time = []

    for trial, trial_evs in word_evs.groupby("trial"):
        delivery_times = np.diff(trial_evs.mstime) / 1000
        total_delivery_time.extend(delivery_times)

    median = np.median(total_delivery_time)
    median = float("{0:.2f}".format(median))
    fig, ax = plt.subplots(figsize=(12,8))
    plt.hist(total_delivery_time, color=(0.4,0.6,0.8), range=(0, 100))
    plt.axvline(np.median(total_delivery_time), color="k")
    plt.title("Navigation time between store visits \n Median: {}s".format(median), fontsize=20)
    plt.xlabel("Time (sec)", fontsize=22)
    plt.ylabel("Count", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.savefig(plot_buffer, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return encode_file(plot_buffer)

def make_navigation_plot(evs, experiment):
    
    word_evs = evs.query('type == "WORD"')
    deliv_table = word_evs.groupby(["subject", "session"]).agg({"trial":pd.Series.nunique}).reset_index()
    

    plots = []
    
    for i, row in deliv_table.iterrows():
        
        plot_buffer = io.BytesIO()
        
        print("{} session {}".format(row.subject, row.session))
        # pull up jsonl file from data10 directory
        data_dir = "/data10/RAM/subjects/"
        file_name = "{}/behavioral/{}/session_{}/session.jsonl".format(row.subject, experiment, row.session)
        file_dir = os.path.join(data_dir, file_name)

        new_file = []
        for line in open(file_dir, "r"):
            # replace this specific entry to empty string
            if '"point condition":SerialPosition,' in line:
                line = line.replace('"point condition":SerialPosition,', '')
            elif '"point condition":SpatialPosition,' in line:
                line = line.replace('"point condition":SpatialPosition,', '')
            elif '"point condition":Random,' in line:
                line = line.replace('"point condition":Random,', '')

            data_dict = json.loads(line)
            new_file.append(data_dict)

        with open("session_tmp.jsonl", "w") as outfile:
            for line in new_file:
                json.dump(line, outfile)
                outfile.write('\n')

        log = pd.read_json("session_tmp.jsonl", lines=True)
        log = log[(log.type=='PlayerTransform')|(log.type=='object presentation begins')]

        def extract_position(row):
            if row.type=='PlayerTransform':
                return (row['data']['positionX'], row['data']['positionY'], row['data']['positionZ'])
            elif row.type=='object presentation begins':
                return eval(row['data']['player position'])
            else:
                return np.nan

        log['location']=log.apply(extract_position, axis=1)
        log['trial']=log.apply(lambda row: row['data']['trial number'] if np.isin('trial number', list(row['data'].keys())) else np.nan, axis=1)
        movements = log.fillna(method='pad').dropna()
        sess_pos = np.stack(list(map(list, movements.location.values)))

        group_mov = movements.groupby('trial')    
        dd_list = np.array(list(group_mov.groups.keys())).astype(int)
        pos_by_list = [np.stack(list(map(list, group_mov.get_group(i)['location']))) for i in dd_list]

        fig, ax = plt.subplots(len(dd_list)//2 + len(dd_list)%2, 2, 
                               figsize= (20, 8*(len(dd_list)//2 + len(dd_list)%2)) )
        ax = ax.ravel()
        fig.suptitle("Session {}".format(row.session), x=0.4, fontsize = 'xx-large', weight='heavy')


        for i, dd in enumerate(dd_list):
            store_locs = log['data'][(log.type=='object presentation begins')&(log.trial==dd)].apply(lambda json: eval(json['store position']))
            store_names = log['data'][(log.type=='object presentation begins')&(log.trial==dd)].apply(lambda json: json['store name'])
            list_stores = np.stack(list(map(list, store_locs.values)))


            points = np.array([pos_by_list[i][:, 0], pos_by_list[i][:, 2]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # don't count motionless time
            segments = segments[1:][np.any(np.diff(segments, axis=0)!=0, axis=(1, 2))]
            lc = LineCollection(segments, cmap='copper_r')
            # Set the values used for colormapping
            lc.set_array(np.linspace(0, 1, len(segments)))
            lc.set_linewidth(3); lc.set_alpha(.6)
            line = ax[i].add_collection(lc)
            ax[i].scatter(list_stores[:, 0], list_stores[:, 2])
            texts = []
            for x, y, name, order in zip(list_stores[:, 0], list_stores[:, 2], store_names, np.arange(len(store_names))):
                texts.append(ax[i].text(x-10, y+5, name.capitalize() + ' ({})'.format(order)))
            ax[i].set_title('Delivery Day %d' %(i+1), fontsize=20)
            ax[i].set_xticklabels([]);ax[i].set_yticklabels([])
        #         adjust_text(texts, expand_text=(1.2, 1.2))


            positions = []
            tmp_texts = []
            for text in texts:
        #             print(text.get_position())
                positions.append(text.get_position())
                tmp_texts.append(text)

            point1_1 = (26.700000000000003, 64.1)
            if point1_1 in positions:
                text1 = tmp_texts[positions.index(point1_1)]
                text1.set_visible(False)
                ax[i].text(26.7, 54.1, text1.get_text())

            if ((25.1, 31.7) in positions) and ((48.3, 32.9) in positions):
                text1 = tmp_texts[positions.index((25.1, 31.7))]
                text2 = tmp_texts[positions.index((48.3, 32.9))]

                text1.set_visible(False); text2.set_visible(False)

                ax[i].text(15.1, 31.7, text1.get_text())
                ax[i].text(52.3, 32.9, text2.get_text())

            point3_1 = (4.5, -28.700000000000003); point3_2 = (25.4, -28.1); point3_3 = (52.3, -28.5)
            if (point3_1 in positions) and (point3_2 in positions):

                if point3_3 in positions:
                    text1 = tmp_texts[positions.index(point3_1)]
                    text2 = tmp_texts[positions.index(point3_2)]
                    text3 = tmp_texts[positions.index(point3_3)]

                    text1.set_visible(False); text2.set_visible(False); text3.set_visible(False)

                    ax[i].text(-10, -28.700000000000003, text1.get_text())
                    ax[i].text(20.4, -28.1, text2.get_text())
                    ax[i].text(61.3, -28.5, text3.get_text())
                else:
                    text1 = tmp_texts[positions.index(point3_1)]
                    text2 = tmp_texts[positions.index(point3_2)]

                    text1.set_visible(False); text2.set_visible(False)

                    ax[i].text(-7, -28.700000000000003, text1.get_text())
                    ax[i].text(25.4, -28.1, text2.get_text())

            elif (point3_2 in positions) and (point3_3 in positions):
                text1 = tmp_texts[positions.index(point3_2)]

                text1.set_visible(False)

                ax[i].text(10.4, -28.1, text1.get_text())

            point4_1 = (-13.1, -53.5); point4_2 = (4.6, -53.8); point4_3 = (22.200000000000003, -55.9)
            if (point4_1 in positions) and (point4_2 in positions):

                if point4_3 in positions:
                    text1 = tmp_texts[positions.index(point4_1)]
                    text2 = tmp_texts[positions.index(point4_2)]
                    text3 = tmp_texts[positions.index(point4_3)]

                    text1.set_visible(False); text2.set_visible(False); text3.set_visible(False)

                    ax[i].text(-20, -65.5, text1.get_text())
                    ax[i].text(0, -53.8, text2.get_text())
                    ax[i].text(30, -55.9, text3.get_text())
                else:
                    text1 = tmp_texts[positions.index(point4_1)]
                    text2 = tmp_texts[positions.index(point4_2)]

                    text1.set_visible(False); text2.set_visible(False)

                    ax[i].text(-20, -53.5, text1.get_text())
                    ax[i].text(5, -53.8, text2.get_text())

            elif (point4_2 in positions) and (point4_3 in positions):
                text1 = tmp_texts[positions.index(point3_2)]

                text1.set_visible(False)

                ax[i].text(0, -53.8, text1.get_text())


        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        plt.tight_layout()
        cbar = fig.colorbar(line, ax=ax[:], location='right', shrink = 0.7)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_yticklabels(["Start", "", "", "", "", "End"])
        fig.set_facecolor('white')
        for a in ax:
            a.axis("off")
        # plt.show()
        plt.savefig(plot_buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        plots.append(encode_file(plot_buffer))
        

    return plots