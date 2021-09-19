import torch
import torch.nn as nn
from datetime import timedelta
from datetime import datetime

n_experiments = 1000
n_data = 15000
channels = 256
height = 32
width = 32

def saynow(text):
    print(f"[{datetime.now()}] {text}") 

avg_duration = timedelta()
max_duration = timedelta()
no_duration = timedelta()

for experiment_index in range(0, n_experiments):
    saynow(f"STARTING EXPERIMENT {experiment_index}/{n_experiments}")
    data = []
    for _ in range(0, n_data):
       data.append(torch.randn(1, channels, height, width).cuda())

    gap_layer = nn.AvgPool2d(kernel_size=(height, width))
    gmp_layer = nn.MaxPool2d(kernel_size=(height, width))

    avgpool_tail = nn.Sequential(gap_layer, nn.Flatten(0, -1), nn.Linear(channels, 10)).cuda()
    maxpool_tail = nn.Sequential(gmp_layer, nn.Flatten(0, -1), nn.Linear(channels, 10)).cuda()
    nopool_tail = nn.Sequential(nn.Flatten(0, -1), nn.Linear(channels * height * width, 10)).cuda()

    avg_start_time = datetime.now()
    for i, datum in enumerate(data):
        result = avgpool_tail(datum)
    avg_end_time = datetime.now()
    avg_duration += (avg_end_time - avg_start_time)

    max_start_time = datetime.now()
    for i, datum in enumerate(data):
        result = maxpool_tail(datum)
    max_end_time = datetime.now()
    max_duration += (max_end_time - max_start_time)

    no_start_time = datetime.now()
    for i, datum in enumerate(data):
        result = nopool_tail(datum)
    no_end_time = datetime.now()
    no_duration += (no_end_time - no_start_time)

n_operations = n_data * n_experiments

saynow(f"AVGPOOL TOTAL TIME: {avg_duration}")
saynow(f"MAXPOOL TOTAL TIME: {max_duration}")
saynow(f"NOPOOL TOTAL TIME: {no_duration}")
saynow(f"AVGPOOL AVERAGE TIME: {avg_duration / n_operations}")
saynow(f"MAXPOOL AVERAGE TIME: {max_duration / n_operations}")
saynow(f"NOPOOL AVERAGE TIME: {no_duration / n_operations}")
