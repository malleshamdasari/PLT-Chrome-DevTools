#!/usr/bin/env python3.5
import json
import pandas as pd
import os
import shutil
from scipy.stats.stats import pearsonr
import subprocess
import time
import itertools
import logging
import coloredlogs
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing

coloredlogs.install(level='DEBUG')
from collections import defaultdict

_experiment_dir = '/home/jnejati/PLTSpeed/desktop_live'
_experiment_dir = '/home/jnejati/PLTSpeed/desktop_live-b1500750d40'

def read_tcptrace_feature(_file, _feature):
    data_array = pd.read_csv(_file, skiprows=8, )
    try:
        host_b = data_array['host_b'].values
    except:
        print(data_array)
        print(_file, _feature)
        exit()
    port_b = data_array['port_b'].values
    feature_list_b2a = data_array[_feature].values
    _total = 0
    for index, my_server in enumerate(host_b):
        if str(feature_list_b2a[index]).strip().isdigit():
            _total = _total + float(feature_list_b2a[index])
        else:
            continue
    return _total

def find_tcpfeature_ratio(f1, f2, _feature):
    _fvalue1 = read_tcptrace_feature(f1, _feature)
    _fvalue2 = read_tcptrace_feature(f2, _feature)
    return _fvalue1 - _fvalue2
 
def find_total_p_r(aList):
    total_pr = 0
    for pr in aList['objs']:
        total_pr += pr[1]['endTime'] - pr[1]['startTime']
    return total_pr

def read_load_time(_file):
    with open(_file) as _f:
        _data = json.load(_f)
    return float(_data[0]['load'])

def find_similarity_toplevel(f1, f2):
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
    _total1 = 0
    _total2 = 0
    for _id1, _event_length1 in data1_dict.items():
        _total1 = _total1 + _event_length1['Networking']
        _total1 = _total1 +  _event_length1['Loading'] + _event_length1['Scripting']
    for _id2, _event_length2 in data2_dict.items():
        _total2 = _total2 + _event_length2['Networking']
        _total2 = _total2 + _event_length2['Loading'] + _event_length2['Scripting']
    _total1 = _total1 + total_rendering1 + total_painting1
    _total2 = _total2 + total_rendering2 + total_painting2
    return round(_total1/_total2, 2)
    """if _total1 < _total2:
        return round(_total1/_total2, 2)
    else:
        return round(_total2/_total1, 2)"""
    #return round(_total1 - _total2, 2)


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


def read_perf(_file, _feature):
    with open(_file):
        _values = pd.read_csv(_file, skiprows=1, header=None).values
        for _value in _values:
            if _feature == _value[2]:
                return float(_value[0])

def read_dns(_file):
    with open(_file) as _f:
        _data = json.load(_f)
    #print(_data[-1].keys())
    return float(_data[-1]['sockets_bytes_in'])
    #return float(_data[-1]['dnsTime'])


def find_feature_ratio(f1, f2, t1, t2, _feature):
    _tvalue1 = read_load_time(t1) 
    _tvalue2 = read_load_time(t2)
    #_fvalue1 = read_perf(f1, _feature)
    #_fvalue2 = read_perf(f2, _feature)
    _fvalue1 = read_dns(t1)
    _fvalue2 = read_dns(t2)
    return _fvalue1, _fvalue2 
    #return (_fvalue1 - _fvalue2)/_fvalue1, (_fvalue1 - _fvalue2)/_fvalue2
    try:
        _ft1 = _fvalue1/_tvalue1
    except:
        print(_feature, f1, f2, _fvalue1, _fvalue2)
        exit()
    _ft2 = _fvalue2/_tvalue2
    return _ft1/_ft2
    if _ft1 > _ft2:
        return _ft1/_ft2
    else:
        return _ft2/_ft1


feature_dict = {}
ratio_dict = {}
perf_feature_list = ['task-clock', 'context-switches', 'branches', 'branch-misses', 'cache-misses', 'cache-references', 'cycles:u',
'cycles:k', 'page-faults', 'sched:sched_switch', 'sched:sched_stat_runtime', 'sched:sched_wakeup', 'instructions:u',
'instructions:k', 'dTLB-load-misses', 'dTLB-loads', 'dTLB-store-misses', 'dTLB-stores', 'iTLB-load-misses', 'iTLB-loads',
'L1-dcache-load-misses', 'L1-dcache-loads', 'L1-dcache-stores', 'L1-icache-load-misses', 'LLC-loads', 'LLC-stores']
perf_feature_list = ['sockets_bytes_in']
#perf_feature_list = ['throughput_b2a']
i= -1
#feature_vector = [[] for i in range(len(perf_feature_list))]
for _feature in perf_feature_list:
    #i += 1
    feature_vector = [[] for i in range(2)]
    label_vector = []
    site_list = []
    for _site_dir in os.listdir(_experiment_dir):
        _site_dir = os.path.join(_experiment_dir, _site_dir)
        _runs = [x for x in os.listdir(_site_dir) if x.startswith('run')]
        _site_name = _site_dir.split('/')[-1]
        for _run_no1 in _runs:
            for _run_no2 in _runs:
                if not _run_no1 == _run_no2:
                    _analysis_dir1 = os.path.join(_site_dir, _run_no1 + '/analysis')
                    _analysis_dir2= os.path.join(_site_dir, _run_no2 + '/analysis')
                    _tcptrace_dir1 = os.path.join(_site_dir, _run_no1 + '/tcptrace')
                    _tcptrace_dir2 = os.path.join(_site_dir, _run_no2 + '/tcptrace')
                    _feature_dir1 = os.path.join(_site_dir, _run_no1 + '/perf')
                    _feature_dir2= os.path.join(_site_dir, _run_no2 + '/perf')
                    if len(os.listdir(_analysis_dir1)) > 0 and len(os.listdir(_analysis_dir2)) > 0:
                        _analysis_file1 = os.path.join(_analysis_dir1, os.listdir(_analysis_dir1)[0])
                        _analysis_file2 = os.path.join(_analysis_dir2, os.listdir(_analysis_dir2)[0])
                        _tcptrace_file1 = os.path.join(_tcptrace_dir1, os.listdir(_tcptrace_dir1)[0])
                        _tcptrace_file2 = os.path.join(_tcptrace_dir2, os.listdir(_tcptrace_dir2)[0])
                        _feature_file1 = os.path.join(_feature_dir1, os.listdir(_feature_dir1)[0])
                        _feature_file2 = os.path.join(_feature_dir2, os.listdir(_feature_dir2)[0])
                    else:
                        #print('missing analysis file in ' + _site_dir.split('/')[-1] + _run_no1 + _run_no2)
                        pass
                    #cur_ratio = find_similarity_ratio(_analysis_file1, _analysis_file2)
                    cur_ratio = find_similarity_toplevel(_analysis_file1, _analysis_file2)
                    _fvalue1, _fvalue2 = find_feature_ratio(_feature_file1, _feature_file2, _analysis_file1, _analysis_file2, _feature)
                    if cur_ratio and _fvalue1 and _fvalue2:
                        label_vector.append(cur_ratio)
                        feature_vector[0].append(_fvalue1)
                        feature_vector[1].append(_fvalue2)
                        site_list.append(_site_name + _run_no1 +  _run_no2)
    feature_vector = np.array(feature_vector).astype(np.float)
    feature_vector = np.transpose(feature_vector)
    #min_max_scaler = preprocessing.MinMaxScaler()
    #feature_vector = min_max_scaler.fit_transform(feature_vector)
    #feature_vector = preprocessing.scale(feature_vector)
    #print(feature_vector.mean(axis=0))
    #print(feature_vector.stdev(axis=0))
    #min_max_scaler = preprocessing.MinMaxScaler()
    #scaled_data = min_max_scaler.fit_transform(feature_vector)

    """print(feature_vector[0][0], label_vector[0])
    print(feature_vector[0][1])
    print(site_list[0])"""
    #regr = linear_model.LinearRegression(fit_intercept=False)
    regr = linear_model.LinearRegression()
    regr.fit(feature_vector, label_vector)
    print('Feature: ' + _feature) 
    print('Coefficients: ', regr.coef_)
    print('R2: ', regr.score(feature_vector, label_vector))
    print('Interecpt: ', regr.intercept_)
    print(100*'-')
