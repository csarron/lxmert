import pickle

with open('data/tx2-yolo-vqa-profile.pk', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    prof_data = pickle.load(f)

timings = dict()
for k, events in prof_data.items():
    if not events:
        continue
    timings[k] = sum([e.total_average().cuda_time_total for e in events])

import csv
with open('data/tx2-yolo-vqa-b1_n10.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for k, v in timings.items():
        writer.writerow(['/'.join(k), v/10000])

from collections import defaultdict
breakdown = defaultdict(list)
for k, v in timings.items():
    if k[1] == 'detection_model':
        breakdown['detection_model'].append(v/1000)
    if '/'.join(k).startswith('VQAModel/lxrt_encoder/model/bert/encoder/layer'):
        breakdown['language_encoder'].append(v/1000)
    if '/'.join(k).startswith('VQAModel/lxrt_encoder/model/bert/encoder/r_layers/'):
        breakdown['vision_encoder'].append(v/1000)
    if '/'.join(k).startswith('VQAModel/lxrt_encoder/model/bert/encoder/x_layers/'):
        breakdown['cross_encoder'].append(v/1000)

for k, v in breakdown.items():
    print(k, sum(v)/10)
