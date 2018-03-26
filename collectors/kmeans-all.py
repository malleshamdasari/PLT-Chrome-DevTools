#!/usr/bin/env python3.5
from collections import Counter
from scipy.spatial import distance
from sklearn.cluster import KMeans
from collections import defaultdict
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

_experiment_dir1 = '/home/jnejati/PLTSpeed/desktop_live'
_experiment_dir2 = '/home/jnejati/PLTSpeed/desktop_live-b1500750d40'
_experiment_dir_list = ['/home/jnejati/PLTSpeed/desktop_live-b1500750d40', '/home/jnejati/PLTSpeed/desktop_live', '/home/jnejati/PLTSpeed/desktop_live-b750250d100']
_experiment_dir_list = [ '/home/jnejati/PLTSpeed/desktop_live', '/home/jnejati/PLTSpeed/desktop_live-b750250d100']
#_experiment_dir_list = ['/home/jnejati/PLTSpeed/desktop_live-b750250d100']


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
            _total += float(feature_list_b2a[index])
        else:
            continue
    return _total


def find_tcpfeature_ratio(f1, f2, _feature):
    _fvalue = read_tcptrace_feature(f1, _feature)
    _fvalue2 = read_tcptrace_feature(f2, _feature)
    return _fvalue - _fvalue2


def find_total_p_r(aList):
    total_pr = 0
    for pr in aList['objs']:
        total_pr += pr[1]['endTime'] - pr[1]['startTime']
    return total_pr

def read_load_time(_file):
    with open(_file) as _f:
        _data = json.load(_f)
    return float(_data[0]['load'])

def find_similarity_reference(_experiment_dir_list):
    _site_dict = {}
    for _experiment_dir in _experiment_dir_list:
        for _site_name in os.listdir(_experiment_dir):
            _site_dir = os.path.join(_experiment_dir, _site_name)
            _site_total_list = []
            data1_dict = defaultdict(dict)
            _runs = os.listdir(_site_dir)
            _runs = [x for x in _runs if x.startswith('run')]
            _runs.sort(key=lambda tup: int(tup.split('_')[1]))
            for _run in _runs:
                _analysis_dir = os.path.join(_site_dir, _run + '/analysis')
                if len(os.listdir(_analysis_dir)) > 0:
                    _analysis_file = os.path.join(_analysis_dir, os.listdir(_analysis_dir)[0])
                else:
                    continue
                with open(_analysis_file) as data_file1:
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
                _total_count = 0
                for _id1, _event_length1 in data1_dict.items():
                    if _event_length1['Networking'] > 0:
                        _total_networking1 = _total_networking1 + _event_length1['Networking']
                        _total_count += 1
                    if _event_length1['Loading'] > 0:
                        _total_loading1 = _total_loading1 + _event_length1['Loading']
                        _total_count += 1
                    if _event_length1['Scripting'] > 0:
                        _total_scripting1 = _total_scripting1 + _event_length1['Scripting']
                        _total_count += 1
                _total = _total_networking1 + _total_loading1 + _total_scripting1 + _total_rendering1 + _total_painting1
                _site_dict.setdefault(os.path.join(_site_dir,  _run), []).append([_total, _total_networking1, _total_loading1, _total_scripting1, _total_rendering1, _total_painting1])
                _site_total_list.append(_total)
            _reference = np.median(_site_total_list)
            _site_dict.setdefault(_site_dir, []).append(_reference)
    return _site_dict

def read_perf(_file, _feature):
    with open(_file):
        _values = pd.read_csv(_file, skiprows=1, header=None).values
        for _value in _values:
            if _feature == _value[2]:
                return float(_value[0])

def read_net(_file, _feature):
    with open(_file) as _f:
        _data = json.load(_f)
    return float(_data[-1][_feature])
    #return float(_data[-1]['dnsTime'])


def find_feature_value(f1, t1, _feature):
    _tvalue = read_load_time(t1) 
    if _feature in perf_feature_list:
        _fvalue = read_perf(f1, _feature)
    elif _feature in net_feature_list:
        _fvalue = read_net(t1, _feature)
    return _fvalue, _tvalue 


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
label_vector = []
time_vector = []
feature_vector = [[] for i in range((1 * len(perf_feature_list)) + (1 * len(net_feature_list)))]
_site_dict = find_similarity_reference(_experiment_dir_list)
_site_dict_key_lookup = {}
_site_names = {}
for _experiment_dir in _experiment_dir_list:
    for _site_name in os.listdir(_experiment_dir):
        _site_dir = os.path.join(_experiment_dir, _site_name)
        _site_names.setdefault(_site_name, []).append(_site_dir)
_site_names = {key: value for key, value in _site_names.items() if len(value) == len(_experiment_dir_list)}

for _site_name, _site_dirs in _site_names.items():
    for _site_dir in _site_dirs:
        _run_nos = os.listdir(_site_dir)
        for _run_no in _run_nos:
            if _run_no.split('/')[-1].startswith('run') and os.path.join(_site_dir, _run_no) in _site_dict.keys():
                _analysis_dir = os.path.join(_site_dir, _run_no, 'analysis')
                _tcptrace_dir = os.path.join(_site_dir, _run_no, 'tcptrace')
                _feature_dir = os.path.join(_site_dir, _run_no, 'perf')
                if len(os.listdir(_analysis_dir)) > 0:
                    _analysis_file = os.path.join(_analysis_dir, os.listdir(_analysis_dir)[0])
                    _tcptrace_file = os.path.join(_tcptrace_dir, os.listdir(_tcptrace_dir)[0])
                    _feature_file = os.path.join(_feature_dir, os.listdir(_feature_dir)[0])
                else:
                    continue
                cur_ratio = round(_site_dict[os.path.join(_site_dir, _run_no)][0][0])
                i = -1
                if cur_ratio:
                    label_vector.append(cur_ratio)
                    _site_dict_key_lookup[len(label_vector) - 1 ] = os.path.join(_site_dir, _run_no)
                for _feature in perf_feature_list + net_feature_list:
                    if cur_ratio:
                        _fvalue, _tvalue = find_feature_value(_feature_file, _analysis_file, _feature)
                        if _fvalue > 0 and _tvalue > 0:
                            i += 1
                            feature_vector[i].append(_fvalue)
                    else:
                        continue
                if cur_ratio and _tvalue > 0:
                    time_vector.append(_tvalue)

np.set_printoptions(suppress=True)
feature_vector = np.array(feature_vector).astype(np.float)
feature_vector = np.transpose(feature_vector)
min_max_scaler = preprocessing.MinMaxScaler()
feature_vector = min_max_scaler.fit_transform(feature_vector)
#feature_vector = preprocessing.scale(feature_vector)
################################################# Kmeans ############################################
n_clusters = 2
km = KMeans(n_clusters)
clusters = km.fit(feature_vector)
labels = clusters.labels_
labels = list(labels)
clusters_by_exp = {}
for i, _label in enumerate(labels):
   _run_no = _site_dict_key_lookup[i]
   _exp_dir = _run_no.split('/')[-3]
   clusters_by_exp.setdefault(_exp_dir, []).append(str(_label))
   print('label: ' + str(_label) + ', run_no: ' + _run_no + ', Total_time: ' + str(_site_dict[_run_no][0][0]))
c = []
with open('./kmeans_results/b1500-live-all-f.txt', 'a') as out_f:
    for k, v in clusters_by_exp.items():
        c = Counter(v)
        out_f.write(_site_name + ': ' + str(k) + ': ' + str(round(max(c.values())/sum(c.values()), 2)) + '\n')
    out_f.write(100*'_' + '\n')
centroids = clusters.cluster_centers_
#print(labels)
#print(centroids)
######################################################################################################

################################################### Regression #######################################
#regr = linear_model.LinearRegression(fit_intercept=False)
"""regr = linear_model.Lasso(alpha=0.1, copy_X=True, fit_intercept=False, max_iter=1000000,
                     normalize=False, positive=False, precompute=False, random_state=None,
                     selection='cyclic', tol=0.0003, warm_start=False)"""
"""print(_site_name)
regr = linear_model.LinearRegression()
regr.fit(feature_vector, label_vector)
#regr.fit(feature_vector, time_vector)
print('Coefficients: ', regr.coef_)
#print('R^2: ', regr.score(feature_vector, time_vector))
print('R^2: ', regr.score(feature_vector, label_vector))
print('Intercept: ', regr.intercept_)
print(100*'-')"""
#######################################################################################################
