#!/usr/bin/env python3.5
import os
import shutil
import subprocess
import time


_experiment_dir = '/home/jnejati/PLTSpeed/desktop_live-b750250d100'
for _site_dir in os.listdir(_experiment_dir):
    _site_dir = os.path.join(_experiment_dir, _site_dir)
    _runs = [x for x in os.listdir(_site_dir) if x.startswith('run')]
    for _run_no in _runs:
        _run_dir = os.path.join(_site_dir, _run_no)
        if _run_no == 'run_0':
            os.system('rm -rf ' + _run_dir)
            time.sleep(1)
