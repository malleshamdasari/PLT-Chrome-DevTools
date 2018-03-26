#!/usr/bin/env python3.5
import os
import shutil
import pickle
import subprocess
import time

def reconstruct_etc_hosts(_dict_file):
    _d_ip_dict = pickle.load(open(_dict_file, "rb" ))
    with open('/etc/hosts', 'w') as _f:
        _f.write('127.0.0.1       localhost  localhost.localdomain PLTSpeed2\n')
        for _domain, sd_ip in _d_ip_dict.items():
            for _subdomain_ip in sd_ip:
                for _subdomain, _ip in _subdomain_ip.items():
                    if _subdomain == '@':
                        _site = _domain
                    else:
                        _site = _subdomain + '.' + _domain
                    _f.write(_ip + '\t\t' + _site + '\n')
                

_experiment_dir = '/home/jnejati/PLTSpeed/desktop_b0201d5-DSL-partial'
_experiment_dir = '/home/jnejati/PLTSpeed/desktop_good3g-10runs'
_experiment_dir = '/home/jnejati/PLTSpeed/desktop_good3g-controlled-300runs'
_experiment_dir = '/home/jnejati/PLTSpeed/desktop_good3g-controlled50runs'
shutil.copy2('/etc/hosts', '/etc/hosts.bak')
for _site_dir in os.listdir(_experiment_dir):
    _site_dir = os.path.join(_experiment_dir, _site_dir)
    _runs = [x for x in os.listdir(_site_dir) if x.startswith('run')]
    _dns_pickle = os.path.join(_site_dir, 'dns/dnsBackup.txt')
    reconstruct_etc_hosts(_dns_pickle)
    time.sleep(2)
    for _run_no in _runs:
        _run_dir = os.path.join(_site_dir, _run_no)
        _tcptrace_dir = os.path.join(_run_dir, 'tcptrace')
        if os.path.isdir(_tcptrace_dir):
            for root, dirs, l_files in os.walk(_tcptrace_dir):
                for f in l_files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
        else:
            os.makedirs(_tcptrace_dir)
        _tcpdump_dir = os.path.join(_run_dir, 'tcpdump')
        for _file in os.listdir(_tcpdump_dir):
            tcptrace_log_file = os.path.join(_tcptrace_dir, str(_file) + "." + 'tcptrace')
            print(tcptrace_log_file)
            time.sleep(1)
            tcptrace_log = open(tcptrace_log_file, 'w+')
            tcptrace_command = ['tcptrace', '-lr', '--csv', os.path.join(_tcpdump_dir, _file)]
            tcptrace = subprocess.Popen(tcptrace_command, stdout=tcptrace_log)
            time.sleep(5)
shutil.copy2('/etc/hosts.bak', '/etc/hosts')
