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
_experiment_dir = '/home/jnejati/PLTSpeed/desktop_live-b750250d100'
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

def find_similarity_reference(_site_dir):
    _site_dict = {}
    _site_total_list = []
    data1_dict = defaultdict(dict)
    _runs = os.listdir(_site_dir)
    _runs = [x for x in _runs if x.startswith('run')]
    _runs.sort(key=lambda tup: int(tup.split('_')[1]))
    for _run in _runs:
        _analysis_dir1 = os.path.join(_site_dir, _run + '/analysis')
        if len(os.listdir(_analysis_dir1)) > 0:
            _analysis_file1 = os.path.join(_analysis_dir1, os.listdir(_analysis_dir1)[0])
        else:
            continue
        with open(_analysis_file1) as data_file1:
            data1 = json.load(data_file1)  # ignoring painting, rendering and deps for now.
            rendering1 = data1[-4]
            painting1 = data1[-3]
            data1 = data1[1:-4]
        _total_rendering1 = find_total_p_r(rendering1)
        _total_painting1 = find_total_p_r(painting1)
        for objs1 in data1:
            _id1 = objs1['id']
            for obj1 in objs1['objs']:
                data1_dict[_id1]['Networking'] = 0.0
                data1_dict[_id1]['Loading'] = 0.0
                data1_dict[_id1]['Scripting'] = 0.0
        for objs1 in data1:
            _id1 = objs1['id']
            for obj1 in objs1['objs']:
                _event_type1 = obj1[0].split('_')[0]
                data1_dict[_id1][_event_type1] += (obj1[1]['endTime'] - obj1[1]['startTime'])
        _total_networking1 = 0
        _total_loading1 = 0
        _total_scripting1 = 0
        _total = 0
        for _id1, _event_length1 in data1_dict.items():
            _total_networking1 = _total_networking1 + _event_length1['Networking']
            _total_loading1 = _total_loading1 + _event_length1['Loading'] 
            _total_scripting1 = _total_scripting1 + _event_length1['Scripting']
        _total = _total_networking1 + _total_loading1 + _total_scripting1 + _total_rendering1 + _total_painting1
        _site_dict.setdefault(_run, []).append([_total, _total_networking1, _total_loading1, _total_scripting1, _total_rendering1, _total_painting1])
        _site_total_list.append(_total)
    return _site_dict, np.median(_site_total_list)

def read_perf(_file, _feature):
    with open(_file):
        _values = pd.read_csv(_file, skiprows=1, header=None).values
        for _value in _values:
            if _feature == _value[2]:
                return float(_value[0])

def read_dns(_file, _feature):
    with open(_file) as _f:
        _data = json.load(_f)
    return float(_data[-1][_feature])
    #return float(_data[-1]['dnsTime'])


def find_feature_value(f1, t1, _feature):
    _tvalue1 = read_load_time(t1) 
    if _feature in perf_feature_list:
        _fvalue1 = read_perf(f1, _feature)
    elif _feature in net_feature_list:
        _fvalue1 = read_dns(t1, _feature)
    return _fvalue1


feature_dict = {}
ratio_dict = {}
perf_feature_list = ['task-clock', 'context-switches', 'branches', 'branch-misses', 'cache-misses', 'cache-references', 'cycles:u',
'cycles:k', 'page-faults', 'sched:sched_switch', 'sched:sched_stat_runtime', 'sched:sched_wakeup', 'instructions:u',
'instructions:k', 'dTLB-load-misses', 'dTLB-loads', 'dTLB-store-misses', 'dTLB-stores', 'iTLB-load-misses', 'iTLB-loads',
'L1-dcache-load-misses', 'L1-dcache-loads', 'L1-dcache-stores', 'L1-icache-load-misses', 'LLC-loads', 'LLC-stores']

perf_feature_list = ['task-clock', 'context-switches', 'branches', 'branch-misses', 'cache-misses', 'cache-references', 'cycles:u',
'page-faults', 'sched:sched_switch', 'sched:sched_stat_runtime', 'sched:sched_wakeup', 'instructions:u',
'dTLB-load-misses', 'dTLB-loads', 'dTLB-stores', 'iTLB-loads',
'L1-dcache-load-misses', 'L1-dcache-loads', 'L1-dcache-stores', 'L1-icache-load-misses']

net_feature_list = ['sockets_bytes_in']
#perf_feature_list = ['task-clock']
feature_vector = [[] for i in range((1 * len(perf_feature_list)) + (1 * len(net_feature_list)))]
label_vector = []
exp_dirs = os.listdir(_experiment_dir)
exp_dirs = list(exp_dirs)
np.set_printoptions(suppress=True)
for _site_dir in exp_dirs:
    feature_vector = [[] for i in range((1 * len(perf_feature_list)) + (1 * len(net_feature_list)))]
    label_vector = []
    _site_dir = os.path.join(_experiment_dir, _site_dir)
    _site_name = _site_dir.split('/')[-1]
    print(_site_name)
    _site_dict, _reference = find_similarity_reference(_site_dir)
    count = 0
    for _run_no1 in _site_dict.keys():
        count += 1
        _analysis_dir1 = os.path.join(_site_dir, _run_no1 + '/analysis')
        _tcptrace_dir1 = os.path.join(_site_dir, _run_no1 + '/tcptrace')
        _feature_dir1 = os.path.join(_site_dir, _run_no1 + '/perf')
        if len(os.listdir(_analysis_dir1)) > 0:
            _analysis_file1 = os.path.join(_analysis_dir1, os.listdir(_analysis_dir1)[0])
            _tcptrace_file1 = os.path.join(_tcptrace_dir1, os.listdir(_tcptrace_dir1)[0])
            _feature_file1 = os.path.join(_feature_dir1, os.listdir(_feature_dir1)[0])
        else:
            #print('missing analysis file in ' + _site_dir.split('/')[-1] + _run_no1 + _run_no2)
            pass
        cur_ratio = round(_site_dict[_run_no1][0][0]/_reference, 2)
        #cur_ratio = round(_site_dict[_run_no1][0][0])
        #print(_site_dict[_run_no1][0][0], _reference, cur_ratio)
        i = -1
        if cur_ratio:
            label_vector.append(cur_ratio)
        for _feature in perf_feature_list + net_feature_list:
            if cur_ratio:
                _fvalue1 = find_feature_value(_feature_file1, _analysis_file1, _feature)
                if _fvalue1 > 0:
                    i += 1
                    feature_vector[i].append(_fvalue1)
            else:
                continue
        """if count == 3: 
            feature_vector = np.array(feature_vector).astype(np.float)
            print(feature_vector.ndim)
            print(feature_vector)
            print(100*'-')
            print(label_vector)
            print(_site_name)
            exit()"""

    np.set_printoptions(suppress=True)
    feature_vector = np.array(feature_vector).astype(np.float)
    feature_vector = np.transpose(feature_vector)
    #min_max_scaler = preprocessing.MinMaxScaler()
    #feature_vector = min_max_scaler.fit_transform(feature_vector)
    feature_vector = preprocessing.scale(feature_vector)
    regr = linear_model.LinearRegression()
    regr.fit(feature_vector, label_vector)
    #print(_site_name)
    print('Coefficients: ', regr.coef_)
    print('R^2: ', regr.score(feature_vector, label_vector))
    print('Intercept: ', regr.intercept_)
    print(100*'-')
         
"""feature_vector = np.array(feature_vector).astype(np.float)
feature_vector = np.transpose(feature_vector)
print(feature_vector.shape)
print(feature_vector.ndim)
print(len(label_vector))
min_max_scaler = preprocessing.MinMaxScaler()
feature_vector = min_max_scaler.fit_transform(feature_vector)
#feature_vector = preprocessing.scale(feature_vector)

#regr = linear_model.LinearRegression(fit_intercept=False)
regr = linear_model.Lasso(alpha=0.1, copy_X=True, fit_intercept=False, max_iter=1000000,
                     normalize=False, positive=False, precompute=False, random_state=None,
                     selection='cyclic', tol=0.0003, warm_start=False)
regr = linear_model.LinearRegression()
regr.fit(feature_vector, label_vector)
print('Feature: ' + _feature) 
print('Coefficients: ', regr.coef_)
print('R2: ', regr.score(feature_vector, label_vector))
print('Interecpt: ', regr.intercept_)
print(100*'-')"""
