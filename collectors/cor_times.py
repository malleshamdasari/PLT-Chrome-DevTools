#!/usr/bin/env python3.5
import os
import json
import shutil
import subprocess
import statistics
import pandas as pd
import itertools
import logging
import coloredlogs
from collections import defaultdict
from scipy.stats.stats import pearsonr   
coloredlogs.install(level='DEBUG')
_similarity_dict  = defaultdict(dict)
_ttfb_dict  = defaultdict(dict)
_tcp_dict  = defaultdict(dict)
 
_experiment_dir = '/home/jnejati/PLTSpeed/desktop_b0201d5-DSL-partial'
for _site_dir in os.listdir(_experiment_dir):
    _similarity_run_dict = {}
    _ttfb_run_dict = {}
    _tcp_run_dict = {}
    _load_run_dict = {}
    _load_list = []
    _ttfb_list = []
    _site_dir = os.path.join(_experiment_dir, _site_dir)
    _runs = [x for x in os.listdir(_site_dir) if x.startswith('run')]
    with open(os.path.join(_site_dir, 'pairwise_normalized.txt')) as _s:
         for line in _s:
             line = line.strip()
             _sim_index = line.split()[2]
             _run1 = line.split()[0]
             _run2 = line.split()[1]
             _similarity_run_dict[_run1 + _run2] =  float(_sim_index)
    _similarity_dict[_site_dir] =_similarity_run_dict
    for _run_no in _runs:
        _run_dir = os.path.join(_site_dir, _run_no)
        _analysis_dir = os.path.join(_run_dir, 'analysis')
        _tcptrace_dir = os.path.join(_run_dir, 'tcptrace')
        for _file in os.listdir(_analysis_dir):
            analysis_file = os.path.join(_analysis_dir, _file)
            with open(analysis_file) as _f:
                _data = json.load(_f)
                _load_list.append((_data[1]['load']))
                _ttfb_run_dict[_run_no] = _data[3]['objs'][0][1]['responseReceivedTime']
                _load_run_dict[_run_no] = _data[1]['load']
        for _tfile in os.listdir(_tcptrace_dir):
            tcptrace_file = os.path.join(_tcptrace_dir, _tfile)
            data_array = pd.read_csv(tcptrace_file, skiprows=8, )
            host_b = data_array['host_b'].values
            port_b = data_array['port_b'].values
            _tcp_run_dict[_run_no] = sum(data_array['total_packets_b2a'].values)
    _ttfb_dict[_site_dir]= _ttfb_run_dict
    _tcp_dict[_site_dir] = _tcp_run_dict


_ttfb_dict_new = {}
_tcp_dict_new = {}
for _site_dir in os.listdir(_experiment_dir):
    _site_dir = os.path.join(_experiment_dir, _site_dir)
    _runs = [x for x in os.listdir(_site_dir) if x.startswith('run')]
    _runs_combinations = [ x for x in (itertools.combinations(_runs, 2)) if x[0] != x[1]]
    _ttfb_pair_dict = {}
    _tcp_pair_dict = {}
    for _run_no_pair in _runs_combinations:
        _ttfb_pair_dict[''.join(list(_run_no_pair))] = [_ttfb_dict[_site_dir][_run_no_pair[0]], _ttfb_dict[_site_dir][_run_no_pair[1]]]
        _tcp_pair_dict[''.join(list(_run_no_pair))] = [_tcp_dict[_site_dir][_run_no_pair[0]], _tcp_dict[_site_dir][_run_no_pair[1]]]
    _ttfb_dict_new[_site_dir] = _ttfb_pair_dict
    _tcp_dict_new[_site_dir] = _tcp_pair_dict

a = []
b = []
t1 = []
t2 = []
logging.info('Correlation between TTFB difference and similarity index')
for _site, _run_ttfb_pair  in _ttfb_dict_new.items(): 
    for _runs, _ttfb in _run_ttfb_pair.items():
        #print(_runs + ' ' + str(_ttfb) + ' ' + str(_similarity_dict[_site][_runs]))
        a.append(abs(int(_ttfb[0]) -  int(_ttfb[1])))
        b.append(_similarity_dict[_site][_runs])
    print(_site.split('/')[-1],  round(pearsonr(a,b)[0], 2))   
logging.info('Correlation between TCP total packets and similarity index')
for _site, _run_tcp_pair  in _tcp_dict_new.items():
    for _runs, _tcp in _run_tcp_pair.items():
        t1.append(abs(int(_tcp[0]) -  int(_tcp[1])))
        t2.append(_similarity_dict[_site][_runs])
    print(_site.split('/')[-1],  round(pearsonr(t1,t2)[0], 2))
