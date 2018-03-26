#!/usr/bin/env python3.5
import json
import os
import shutil
import subprocess
import time
import itertools
import logging
import coloredlogs
coloredlogs.install(level='DEBUG')
from collections import defaultdict

_experiment_dir = '/home/jnejati/PLTSpeed/desktop_live-b1500750d40'

def find_total_p_r(aList):
    total_pr = 0
    for pr in aList['objs']:
        total_pr += pr[1]['endTime'] - pr[1]['startTime']
    return total_pr


def find_similarity_ratio(f1, f2):
    ratio = 0.0
    data1_dict = defaultdict(dict)
    data2_dict = defaultdict(dict)
    with open(f1) as data_file1:
        data1 = json.load(data_file1) # ignoring painting, rendering and deps for now.
        rendering1 = data1[-4]
        painting1 = data1[-3]
        data1 = data1[1:-4]
    with open(f2) as data_file2:
        data2 = json.load(data_file2)
        rendering2 = data2[-4]
        painting2 = data2[-3]
        data2 = data2[1:-4]
    total_rendering1 = find_total_p_r(rendering1)
    total_rendering2 = find_total_p_r(rendering2)
    total_painting1 = find_total_p_r(painting1)
    total_painting2 = find_total_p_r(painting2)
    for objs1 in data1:
        _id1 = objs1['id']
        for obj1 in objs1['objs']:
            data1_dict[_id1]['Networking'] = 0.0
            data1_dict[_id1]['Loading'] = 0.0
            data1_dict[_id1]['Scripting'] = 0.0
    for objs2 in data2:
        _id2 = objs2['id']
        for obj2 in objs2['objs']:
            data2_dict[_id2]['Networking'] = 0.0
            data2_dict[_id2]['Loading'] = 0.0
            data2_dict[_id2]['Scripting'] = 0.0
    for objs1 in data1:
        _id1 = objs1['id']
        for obj1 in objs1['objs']:
            _event_type1 = obj1[0].split('_')[0]
            data1_dict[_id1][_event_type1] += (obj1[1]['endTime'] - obj1[1]['startTime']) 
    for objs2 in data2:
        _id2 = objs2['id']
        for obj2 in objs2['objs']:
            _event_type2 = obj2[0].split('_')[0]
            data2_dict[_id2][_event_type2] += (obj2[1]['endTime'] - obj2[1]['startTime']) 
    _total_event_count = 0
    for _id1, _event_length1 in data1_dict.items():
        if _id1 in data2_dict:
            for _event1, _length1 in _event_length1.items():
                if _event1 in data2_dict[_id1]:
                    _length2 = data2_dict[_id1][_event1]
                    if _length1 > 0 and _length2 > 0:
                        _denom = max(_length1, _length2)
                        ratio += round(abs(_length1 - _length2) / _denom, 2)
                        _total_event_count += 1

    rendering_ratio = round(abs(total_rendering1 - total_rendering2) / max(total_rendering1, total_rendering2), 2)
    _total_event_count += 1
    painting_ratio = round(abs(total_painting1 - total_painting2) / max(total_painting1, total_painting2), 2)
    _total_event_count += 1
    return round(ratio/_total_event_count, 2)


for _site_dir in os.listdir(_experiment_dir):
    _site_dir = os.path.join(_experiment_dir, _site_dir)
    _runs = [x for x in os.listdir(_site_dir) if x.startswith('run')]
    _runs_combinations = [ x for x in (itertools.combinations(_runs, 2)) if x[0] != x[1]]
    with open (os.path.join(_site_dir, 'pairwise_normalized.txt'), 'w') as _f:
        for _run_no_pair in _runs_combinations:
            _analysis_dir1 = os.path.join(_site_dir, _run_no_pair[0] + '/analysis')
            _analysis_dir2= os.path.join(_site_dir, _run_no_pair[1] + '/analysis')
            if len(os.listdir(_analysis_dir1)) > 0 and len(os.listdir(_analysis_dir2)) > 0:
                _analysis_file1 = os.path.join(_analysis_dir1, os.listdir(_analysis_dir1)[0])
                _analysis_file2 = os.path.join(_analysis_dir2, os.listdir(_analysis_dir2)[0])
            else:
                continue
            logging.info('Caclulating similarity for ' + _site_dir.split('/')[-1] + ' -- ' + str(_run_no_pair))
            cur_ratio = find_similarity_ratio(_analysis_file1, _analysis_file2) 
            if cur_ratio:
                _f.write(str( _run_no_pair[0]) + '\t' + str(_run_no_pair[1]) + '\t' + str(cur_ratio) + '\n')
   
