#!/usr/bin/python3.5
import subprocess
import time
import os

_experiment_dir = '/home/jnejati/PLTSpeed/desktop_live-b1500750d40'
_experiment_dir = '/home/jnejati/PLTSpeed/desktop_live-good3g'
_experiment_dir = '/home/jnejati/PLTSpeed/desktop_live-good3g-300runs'
_experiment_dir = '/home/jnejati/PLTSpeed/desktop_good3g-10runs'
_experiment_dir = '/home/jnejati/PLTSpeed/desktop_good3g-controlled-300runs'
_experiment_dir = '/home/jnejati/PLTSpeed/desktop_good3g-controlled50runs'
for _site_dir in os.listdir(_experiment_dir):
    _site_dir = os.path.join(_experiment_dir, _site_dir)
    _runs = [x for x in os.listdir(_site_dir) if x.startswith('run')]
    for _run_no in _runs:
        _run_dir = os.path.join(_site_dir, _run_no)
        _justniffer_dir = os.path.join(_run_dir, 'justniffer')
        if os.path.isdir(_justniffer_dir):
            for root, dirs, l_files in os.walk(_justniffer_dir):
                for f in l_files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
        else:
            os.makedirs(_justniffer_dir)
        _tcpdump_dir = os.path.join(_run_dir, 'tcpdump')
        for _file in os.listdir(_tcpdump_dir):
            justniffer_log_file = os.path.join(_justniffer_dir, str(_file) + "." + 'justniffer')
            print(justniffer_log_file)
            time.sleep(1)
            justniffer_log = open(justniffer_log_file, 'w+')
            justniffer_command = ['justniffer', '-f', os.path.join(_tcpdump_dir, _file), '-a', '%connection.time %idle.time.0 %request.time %response.time %response.time.begin %response.time.end %idle.time.1']
            justniffer = subprocess.Popen(justniffer_command, stdout=justniffer_log)
