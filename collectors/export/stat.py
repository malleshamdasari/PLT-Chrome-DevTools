#!/usr/bin/python3.5
import pandas as pd
import os
import statistics

_experiment_dir = '/home/jnejati/PLTSpeed/collectors/export/desktop_good3g-controlled50runs'
for _file in os.listdir(_experiment_dir):
    _file = os.path.join(_experiment_dir, _file)
    _data = pd.read_csv(_file)
    _mean = statistics.mean(_data[' time '])
    _stdev = statistics.stdev(_data[' time '])
    print(_file.split('/')[-1].split('.csv')[0] + '--> mean: ' + str(_mean) + ' stdev: ' +  str(round(_stdev, 2))  + ' stdev/mean: ' + str(round(_stdev/_mean, 2)))

