__author__ = 'jnejati'

import os
from os import path as p
import json
from scipy.stats.stats import pearsonr
import numpy as np

def read_sumary_file(_file, _feature, _metric):
    with open(_file) as _f:
        _data = json.load(_f)
    for _obj in _data:
        if _feature in _obj:
            #print(_file, _feature, _metric, str(float(_obj[_feature][_metric])))
            return float(_obj[_feature][_metric])
    return None
 
def read_load_time(_file):
    with open(_file) as _f:
        _data = json.load(_f)
    return float(_data[1]['load'])

def read_ttfb_time(_file):
    with open(_file) as _f:
        _data = json.load(_f)
    return float(_data[1]['objs'][0][1]['responseReceivedTime'])


def main():
    _experiment_dir = '/home/jnejati/PLTSpeed/desktop_b0201d5-DSL-partial'
    _sample_summary_f = '/home/jnejati/PLTSpeed/plotting/conf/summary.json'
    _feature_list = []
    _cor_dict = {}
    with open(_sample_summary_f) as _s:
        _d = json.load(_s)
        for _jobs in _d:
            for _feature in _jobs:
                _feature_list.append(_feature)
    _sites = os.listdir(_experiment_dir)
    _sites.sort()
    _metric_list = ['sum', 'count']
    _metric_list = ['sum']
    for _metric in _metric_list:
        for _feature in _feature_list:
            _cor_list = []
            for _site_dir in _sites:
                _site_dir = os.path.join(_experiment_dir, _site_dir)
                _runs = [x for x in os.listdir(_site_dir) if x.startswith('run')]
                _runs.sort(key=lambda tup: int(tup.split('_')[1]))
                _site_name = _site_dir.split('/')[-1]
                _feature_value = []
                _load_value = []
                for _run_no in _runs:
                    _run_dir = os.path.join(_site_dir, _run_no)
                    _summary_dir = os.path.join(_run_dir, 'summary')
                    _analysis_dir = os.path.join(_run_dir, 'analysis')
                    _sfile = os.listdir(_summary_dir)
                    _sfile = [ x for x in _sfile if x.endswith('json')]
                    _afile = os.listdir(_analysis_dir)
                    if len(_sfile) == 1 and len(_afile) == 1:
                        _summary_file = os.path.join(_summary_dir, _sfile[0])
                        _f = read_sumary_file(_summary_file, _feature, _metric)
                        _analysis_file = os.path.join(_analysis_dir, _afile[0])
                        _load = read_load_time(_analysis_file)
                        _load_value.append(_load)
                        if _f:
                            _feature_value.append(_f/_load)
                        else:
                            continue
                    else:
                        continue
                    #_ttfb_value.append(read_ttfb_time(_analysis_file))
                try:
                    if len(_feature_value) > 5:
                        _cor = pearsonr(_feature_value, _load_value)[0]
                        _cor_list.append(_cor)
                    else:
                        print('less than 5 ' + str(_site_dir.split('/')[-1]) + str(_feature_value))
                        continue   
                except:
                    #print('Exception')
                    #print(_site_name, _feature,  _feature_value, _cor)
                    continue
            _cor_dict[_feature] =  _cor_list
    with open('/home/jnejati/PLTSpeed/collectors/cor_result.txt', 'w+') as _outf:
        for k,v in _cor_dict.items():
            if len(v)> 0:
                _v= [abs(x) for x in v]
                _c = 0
                for _item in _v:
                    if _item > 0.50:
                        _c += 1
                if _c >= 0.35 * len(v):
                    _outf.write(str(k) + '\t' + str(_c/len(_v)) + '\n') 

if __name__ == "__main__":
    main()


