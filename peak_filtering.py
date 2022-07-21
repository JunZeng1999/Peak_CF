import pandas as pd
import numpy as np

mz = []
location = []
# copy the result table to the current directory
# result table of blank group detected by Peak_CF
csv_data1 = pd.read_csv('your_result_name1.csv')
mz_mean1 = csv_data1['row m/z'].tolist()
rt_mean1 = csv_data1['Peak RT start'].tolist()
rt_mean1_1 = csv_data1['Peak RT end'].tolist()
ph_mean1 = csv_data1['Peak height'].tolist()
mz_mean1 = np.array(mz_mean1)
rt_mean1 = np.array(rt_mean1)
rt_mean1_1 = np.array(rt_mean1_1)
ph_mean1 = np.array(ph_mean1)
# result table of administration group detected by Peak_CF
csv_data2 = pd.read_csv('your_result_name2.csv')
mz_mean2 = csv_data2['row m/z'].tolist()
rt_mean2 = csv_data2['row retention time'].tolist()
ph_mean2 = csv_data2['Peak height'].tolist()
mz_mean2 = np.array(mz_mean2)
rt_mean2 = np.array(rt_mean2)
ph_mean2 = np.array(ph_mean2)
for i in range(len(mz_mean2)):
    for j in range(len(mz_mean1)):
        a = mz_mean2[i] - mz_mean1[j]
        if abs(a) <= 0.005:
            bb = rt_mean2[i]
            bbb = bb - rt_mean1[j]
            bbbb = bb - rt_mean1_1[j]
            if bbbb <= 0.05 and bbb >= -0.05:
                c = ph_mean2[i]/ph_mean1[j]
                if 0.3 < c < 3:
                    location.append(i)
csv_data2 = csv_data2.drop(location)
# table of the possible metabolite peaks
csv_data2.to_csv("your_result_name3.csv", index=False, encoding="utf-8")
