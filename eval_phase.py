import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.pyplot import MultipleLocator
# from os import path

name = 'data/08_30_2022_13_40_11_rtls_raw_iq_samples_f46077833421_0_loop1'
df = pd.read_csv(name + '.csv')
pkt2 = max(df.pkt) + 1
antenna_spacing = 0.28  # Unit: λ
sample_slot = df.iloc[0].slot_duration
sample_rate = df.iloc[0].sample_rate
sample_count = (8 + 37 * 2 // sample_slot) * sample_rate
print('Packet: %d\nSample Count: %d' % (pkt2, sample_count))
PI = np.pi
es_aoa_all = []
better_aoa_list = []
avg_data_score = 0


def corr_phase_dire(phase_diff):
    if phase_diff > PI:
        return phase_diff - 2*PI
    elif phase_diff < -PI:
        return phase_diff + 2*PI
    return phase_diff


for pkt in range(pkt2):
    iq_array = df[df.pkt == pkt][df.columns[5:7]].to_numpy()
    phase_array = []
    for (i, q) in iq_array:
        phase_array.append(np.arctan2(q, i))
    phase_diff_array = []
    for k in range((sample_count//sample_rate-8)//3):
        for l in range(sample_rate):
            phase_diff_array.append(corr_phase_dire(
                phase_array[l+(3*k+8)*sample_rate]-phase_array[l+(3*k+9)*sample_rate]))
            phase_diff_array.append(corr_phase_dire(
                phase_array[l+(3*k+9)*sample_rate]-phase_array[l+(3*k+10)*sample_rate]))
    data_score = 100 - np.std(phase_diff_array)/(2*PI*antenna_spacing)*100
    avg_data_score = (avg_data_score * pkt + data_score) / (pkt + 1)
    sin_aoa = np.mean(phase_diff_array)/(2*PI*antenna_spacing)
    sin_aoa = min(1, max(sin_aoa, -1))
    es_aoa = np.arcsin(sin_aoa)/PI*180
    es_aoa_all.append(es_aoa)
    if data_score > 80:
        better_aoa_list.append(es_aoa)


print('Average Date Quality: %.1f\nAverage AoA: %.1f°\nBetter Rate: %.2f%%\nBetter AoA: %s°' %
      (avg_data_score, np.mean(es_aoa_all), len(better_aoa_list) / len(es_aoa_all) * 100, np.mean(better_aoa_list) if len(better_aoa_list) else 'NaN'))

plt.hist(es_aoa_all, edgecolor='k')
plt.xlabel('Estimated AoA/°')
plt.ylabel('Number')
plt.title('Distribution of Estimated AoA')
plt.show()
