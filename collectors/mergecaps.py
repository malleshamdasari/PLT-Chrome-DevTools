import os
import subprocess
####
### ATTENTION RUN NO needs to be set
####
base_path = '/home/jnejati/PLTSpeed'
site_list =['www.alexa.com', 'www.apple.com',  'www.msn.com','www.newrepublic.com', 'www.thinkprogress.org', 'www.craigslist.org', 'www.townhall.com', 'www.wikipedia.org']
#site_list =['www.alexa.com', 'www.apple.com', 'www.msn.com']
#site_list = ['www.buzzfeed.com', 'www.collegehumor.com', 'www.imdb.com', 'www.indianexpress.com', 'www.irs.gov', 'www.kbb.com', 'www.zdnet.com']
path_prefix = 'desktop_average_4g_compressed_orig_b0200d400_'
 
runs = 50
for cur_site in site_list:
    cur_dir = os.path.join(base_path, path_prefix + cur_site, 'net_logs')
    cur_files = os.listdir(cur_dir)
    output_dir = os.path.join(base_path, path_prefix + cur_site, 'net_logs_merged')
    for i in range(runs):
        same_run = []
        for c_file in cur_files:
            run_no = c_file.split('_')[0]
            if int(run_no) == i:
                same_run.append(os.path.join(cur_dir, c_file))
        output_file = os.path.join(output_dir, str(i) + '_' + cur_site + '-merged.tcpdump')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        command = ['mergecap', '-w', output_file] + same_run
        subprocess.call(command)
