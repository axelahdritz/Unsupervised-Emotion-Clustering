import pandas as pd
import json
import os
import io
import csv
import datetime


def get_real_time(beg_pos, cl, time, t):
    for ind in range(beg_pos, len(t)):
        if cl == 0:
            st, en = 2, 3
        else:
            st, en = 4, 5
        if t[ind][st] <= time <= t[ind][en]:
            if t[ind][st] != t[ind][en]:
                diff = time - t[ind][st]
                real_time = t[ind][0] + diff
                return ind, real_time


csv_files = os.listdir("transcripts/sentence_transcripts/")
filenames = [f[9:19] for f in csv_files if f.endswith('.csv')]

grouping = []
for f in filenames:
    files_to_process = []
    files_to_process.append(f)
    for cs in csv_files:
        if f in cs:
            files_to_process.append(cs)
    grouping.append(files_to_process)

column_names = ["prompt","completion"] 
df = pd.DataFrame(columns = column_names)

with open("complete_segments.json", "r") as jsonfile:
    segments = json.load(jsonfile)

directory = "transcripts/sentence_transcripts/"

# MAIN
for group in grouping:
    if len(group)<3:
        pass
    if group[1][-5] == "s":
        me = os.path.join(directory,group[2])
        others = os.path.join(directory,group[1])
    else:
        me = os.path.join(directory,group[1])
        others = os.path.join(directory,group[2])
    filename = group[0] + ".wav"
    for i in segments:
        if i[0][-14:] == filename:
            segment_list = i[1]
            classes = i[2]

    df_m = pd.read_csv(me)
    df_o = pd.read_csv(others)

    idx_m = pd.to_timedelta(df_m[df_m.columns[3]]).dt.total_seconds()
    idx_mend = pd.to_timedelta(df_m[df_m.columns[4]]).dt.total_seconds()
    me_times = pd.concat([idx_m, idx_mend], axis = 1) 

    idx_o = pd.to_timedelta(df_o[df_o.columns[3]]).dt.total_seconds()
    idx_oend = pd.to_timedelta(df_o[df_o.columns[4]]).dt.total_seconds()
    others_times = pd.concat([idx_o, idx_oend], axis = 1)

    t = []
    i = 0
    for seg in segment_list:
        if i == 0:
            splits = [seg[0], seg[1], 0, 0, 0, 0]
        else:
            splits = [seg[0], seg[1], t[-1][3], t[-1][3], t[-1][5], t[-1][5]]

        diff = seg[1] - seg[0]
        if diff < 1:
            splits[3] += diff
            splits[5] += diff
        
        elif classes[i] == 0:
            splits[3] += diff
        
        elif classes[i] == 1:
            splits[5] += diff
        t.append(splits)
        i += 1
    
    real_time = []
    s = 0
    for index, row in me_times.iterrows():
        s, real_start = get_real_time(s, 0, row['start_time'], t)
        s, real_end = get_real_time(s, 0, row['end_time'], t)
        real_time.append([0,index, real_start, real_end])

    s = 0
    for index, row in others_times.iterrows():
        s, real_start = get_real_time(s, 1, row['start_time'], t)
        s, real_end = get_real_time(s, 1, row['end_time'], t)
        real_time.append([1, index, real_start, real_end])

    times = pd.DataFrame(real_time, columns = ["class", "original_index", "real_start", "real_end"])
    sorted_times = times.sort_values(by =["real_start"])

    i = 0
    prev_end_time = -1
    prev_class = -1
    prompt = ''
    completion = ''
    for i, row in sorted_times.iterrows():
        ind = row["original_index"]
        cl = row["class"]
        if i == 0 and cl == 0:
            prompt = "START:"
            completion = df_m["sentence"].values[int(ind)]
            df.loc[len(df.index)] = [prompt, completion]
            prompt = ''
            completion = ''
        elif prev_class == 0 and cl == 1:
            df.loc[len(df.index)] = [prompt, completion]
            prompt = df_o["sentence"].values[int(ind)]
            completion = ''
        elif prev_class == 0 and cl == 0:
            completion = completion + " " + df_m["sentence"].values[int(ind)]
        elif prev_class == 1 and cl == 1:
            prompt = prompt + " " + df_o["sentence"].values[int(ind)] 
        elif prev_class == 1 and cl == 0:
            completion = df_m["sentence"].values[int(ind)]
        
        prev_end_time = row["real_end"]
        if i == 0:
            prev_class = -1
        else:
            prev_class = cl
        i+=1

df.to_csv("transcripts/complete_conversation.csv", index=True)
