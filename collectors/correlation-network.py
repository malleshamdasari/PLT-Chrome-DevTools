__author__ = 'jnejati'

import os
import statistics
import json
from scipy.stats.stats import pearsonr
import numpy as np
import pandas as pd
import math

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

def read_tcptrace_sum(_file):
   data_array = pd.read_csv(_file, skiprows=8, )
   host_b = data_array['host_b'].values
   port_b = data_array['port_b'].values
   return sum(data_array['total_packets_b2a'].values)


def read_tcptrace_throughput_sum(_file):
   data_array = pd.read_csv(_file, skiprows=8, )
   _b2_a = [int(x) for x in data_array['throughput_b2a'].values[:-1] if str(x).isdigit()]
   #_b2_a = _b2_a.astype(int)
   try:
       return sum(_b2_a)
   except:
       print(_file)
       print(type(_b2_a), _b2_a)
       for v in _b2_a:
           print(v, type(v))
       exit()


def read_tcptrace_throughput_max(_file):
   data_array = pd.read_csv(_file, skiprows=8, )
   _b2_a = [int(x) for x in data_array['throughput_b2a'].values[:-1] if str(x).isdigit()]
   if len(_b2_a) > 0:
       try: 
           return max(_b2_a)
       except:
           print(_file)
           print(type(_b2_a))
           for v in _b2_a:
               print(v, type(v))
           exit()
   else:
       return None 

def read_tcptrace_throughput_avg(_file):
   data_array = pd.read_csv(_file, skiprows=8, )
   _b2_a = [int(x) for x in data_array['throughput_b2a'].values[:-1] if str(x).isdigit()]
   if len(_b2_a) > 0:
       try:
           print(sum(_b2_a)/len(_b2_a))
           return sum(_b2_a)/len(_b2_a)
       except:
           print(_file)
           print(type(_b2_a))
           for v in _b2_a:
               print(v, type(v))
           exit()
   else:
       return None

def read_tcptrace_stdev(_file):
   data_array = pd.read_csv(_file, skiprows=8, )
   tcp_list = data_array['total_packets_b2a'].values
   return  statistics.stdev(tcp_list)


def main():
    _experiment_dir = '/home/jnejati/PLTSpeed/desktop_live'
    _sites = os.listdir(_experiment_dir)
    _sites.sort()
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
             _analysis_dir = os.path.join(_run_dir, 'analysis')
             _tcptrace_dir = os.path.join(_run_dir, 'tcptrace')
             _afile = os.listdir(_analysis_dir)
             _tfile = os.listdir(_tcptrace_dir)
             if len(_tfile) == 1 and len(_afile) == 1:
                 _tcptrace_file = os.path.join(_tcptrace_dir, _tfile[0])
                 _analysis_file = os.path.join(_analysis_dir, _afile[0])
                 _load = read_load_time(_analysis_file)
                 _f = read_tcptrace_throughput_avg(_tcptrace_file)
                 if _f:
                     _feature_value.append(_f/_load)
                     _load_value.append(_load)
                     
         #print(_feature_value, _load_value)
         _cor = pearsonr(_feature_value, _load_value)[0]
         #print(_site_dir.split('/')[-1], _cor)
         if abs(_cor) and not math.isnan(_cor):
             _cor_list.append(abs(_cor))
         #print(100*'_')

    print('Average: ' + str(sum(_cor_list) /len( _cor_list)))


if __name__ == "__main__":
    main()


