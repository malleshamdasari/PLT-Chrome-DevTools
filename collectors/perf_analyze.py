import os
import shutil
import subprocess


data_folder_list = ['desktop_average_4g_compressed_orig_b0200d400_www.alexa.com',
                    'desktop_average_4g_compressed_orig_b0200d400_www.apple.com',
                    'desktop_average_4g_compressed_orig_b0200d400_www.craigslist.org',
                    'desktop_average_4g_compressed_orig_b0200d400_www.msn.com',
                    'desktop_average_4g_compressed_orig_b0200d400_www.newrepublic.com',
                    'desktop_average_4g_compressed_orig_b0200d400_www.thinkprogress.org',
                    'desktop_average_4g_compressed_orig_b0200d400_www.townhall.com',
                    'desktop_average_4g_compressed_orig_b0200d400_www.wikipedia.org']

"""data_folder_list = ['desktop_average_4g_compressed_orig_b2d100_www.buzzfeed.com',
                    'desktop_average_4g_compressed_orig_b2d100_www.collegehumor.com',
                    'desktop_average_4g_compressed_orig_b2d100_www.imdb.com',
                    'desktop_average_4g_compressed_orig_b2d100_www.indianexpress.com',
                    'desktop_average_4g_compressed_orig_b2d100_www.irs.gov',
                    'desktop_average_4g_compressed_orig_b2d100_www.kbb.com',
                    'desktop_average_4g_compressed_orig_b2d100_www.zdnet.com']"""

for data_folder in data_folder_list:
    path = '/home/jnejati/PLTSpeed/' + data_folder + '/perfsched_logs/'
    out_path = '/home/jnejati/PLTSpeed/' + data_folder + '/perfsched_logs_latency/'
    if os.path.isdir(out_path):
        for root, dirs, l_files in os.walk(out_path):
            for f in l_files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
    else:
        os.makedirs(out_path)
    for roots, dirs, files in os.walk(path):
        for my_file in files:
            if my_file.endswith('.data'):
                my_logfile = open(os.path.join(out_path, my_file).split('.data')[0] + "_latency" + '.data', 'w')
                print(os.path.join(roots, my_file))
                command = ['perf', 'sched', '-i', os.path.join(roots, my_file), 'latency']
                cur_perf = subprocess.call(command, stdout=my_logfile, timeout=30)

