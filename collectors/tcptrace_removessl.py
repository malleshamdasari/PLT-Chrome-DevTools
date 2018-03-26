import os
import shutil
import subprocess
import time

"""data_folder_list = ['desktop_average_4g_inlined_orig_coreonly_www.alexa.com',
                    'desktop_average_4g_inlined_orig_coreonly_www.apple.com',
                    'desktop_average_4g_inlined_orig_coreonly_www.bhwd.me',
                    'desktop_average_4g_inlined_orig_coreonly_www.craigslist.org',
                    'desktop_average_4g_inlined_orig_coreonly_www.msn.com',
                    'desktop_average_4g_inlined_orig_coreonly_www.thinkprogress.org',
                    'desktop_average_4g_inlined_orig_coreonly_www.wikipedia.org']"""
data_folder_list = ['desktop_average_4g_inlined2_orig_coreonly_www.alexa.com',
                     'desktop_average_4g_inlined2_orig_coreonly_www.apple.com',
                     'desktop_average_4g_inlined2_orig_coreonly_www.msn.com']
for data_folder in data_folder_list:
    tcptrace_path = '/home/jnejati/PLTSpeed/' + data_folder + '/tcptrace/'
    for dirpath, dirnames, files in os.walk(tcptrace_path):
        for filename in files:
            i = 0
            tcptrace_log_file = os.path.join(tcptrace_path, filename)
            f = open(tcptrace_log_file, "r")
            lines = f.readlines()
            f.close()
            f = open(tcptrace_log_file, "w")
            for line in lines:
                i+=1
                if i> 10:
                    if not line.split(',')[4] == '443':
                        f.write(line)
                else:
                    f.write(line)
            f.close()
