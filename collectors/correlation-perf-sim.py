#!/usr/bin/python3.5
import os
from os import path as p
import json
from scipy.stats.stats import pearsonr
import numpy as np
import pandas as pd
import logging

def read_load_time(_file):
    with open(_file) as _f:
        _data = json.load(_f)
    return float(_data[0]['load'])

def read_ttfb_time(_file):
    with open(_file) as _f:
        _data = json.load(_f)
    return float(_data[1]['objs'][0][1]['responseReceivedTime'])

def read_perf(_file, _feature):
    _values = pd.read_csv(_file, skiprows=1, header=None).values
    for _value in _values:
        if _feature == _value[2]:
            return float(_value[0])

def read_similarit(_file):
    _values = pd.read_csv(_file, skiprows=1, header=None).values

perf_feature_list = ['task-clock', 'context-switches', 'branches', 'branch-misses', 'cache-misses', 'cache-references', 'cycles:u',
'cycles:k', 'page-faults', 'sched:sched_switch', 'sched:sched_stat_runtime', 'sched:sched_wakeup', 'instructions:u',
'instructions:k', 'dTLB-load-misses', 'dTLB-loads', 'dTLB-store-misses', 'dTLB-stores', 'iTLB-load-misses', 'iTLB-loads',
'L1-dcache-load-misses', 'L1-dcache-loads', 'L1-dcache-stores', 'L1-icache-load-misses', 'LLC-loads', 'LLC-stores']

def main():
    logging.getLogger().setLevel(logging.INFO)
    _experiment_dir = '/home/jnejati/PLTSpeed/desktop_live-b1500750d40'
    _sample_summary_f = '/home/jnejati/PLTSpeed/plotting/conf/summary.json'
    _sites = os.listdir(_experiment_dir)
    _sites.sort()
    _cor_dict = {}
    for _pfeature in perf_feature_list:
        _cor_list = []
        logging.info('Analyzing feature: ' + _pfeature)
        for _site_dir in _sites:
            #logging.info('Site: ' + _site_dir.split('/')[-1])
            _site_dir = os.path.join(_experiment_dir, _site_dir)
            _runs = [x for x in os.listdir(_site_dir) if x.startswith('run')]
            _runs.sort(key=lambda tup: int(tup.split('_')[1]))
            _site_name = _site_dir.split('/')[-1]
            _feature_value = []
            _load_value = []
            for _run_no in _runs:
                _run_dir = os.path.join(_site_dir, _run_no)
                _perf_dir = os.path.join(_run_dir, 'perf')
                _analysis_dir = os.path.join(_run_dir, 'analysis')
                _pfile = os.listdir(_perf_dir)
                _pfile.sort()
                _afile = os.listdir(_analysis_dir)
                _afile.sort()
                if len(_pfile) == 1 and len(_afile) == 1:
                    _perf_file = os.path.join(_perf_dir, _pfile[0])
                    _f = read_perf(_perf_file, _pfeature)
                    _analysis_file = os.path.join(_analysis_dir, _afile[0])
                    _load = read_load_time(_analysis_file)
                    _load_value.append(_load)
                    if _f:
                        _feature_value.append(_f/_load)
                    else:
                        continue
                else: # more than file in dir?
                    continue
            try:
                if len(_feature_value) > 10:
                    _cor = pearsonr(_feature_value, _load_value)[0]
                    _cor_list.append(_cor)
                else:
                    print('less than 10 ' + str(_site_dir.split('/')[-1]) + str(_feature_value))
                    continue
            except:
                #print('Exception')
                #print(_site_name, _feature,  _feature_value, _cor)
                continue
        print(_cor_list)
        exit()
        _cor_dict[_pfeature] = _cor_list
    with open('/home/jnejati/PLTSpeed/collectors/perf_cor_result_70.txt', 'w+') as _outf:
        for k, v in _cor_dict.items():
            if len(v) > 0:
                _v = [abs(x) for x in v]
                _c = 0
                for _item in _v:
                    if _item > 0.70:
                        _c += 1
                #if _c >= 0.40 * len(v):
                _outf.write(str(k) + '\t' + str(_c/len(_v)) + '\n')

if __name__ == "__main__":
    main()
