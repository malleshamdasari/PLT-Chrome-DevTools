import pandas as pd
import numpy as np
import os


class ReadFiles:
    def __init__(self, ftype):
        self.ftype = ftype
        # self.path = path

    def read_perf_data(self, path):
        data_path_list = [None] * 200
        feature_list = []
        feature_vector_dict = {
            'task-clock': True,
            'context-switches': True,
            'branches': True,
            'branch-misses': True,
            'cache-misses': True,
            'cache-references': True,#
            'cycles:u': True,
            'cycles:k': True, #
            'page-faults': True,
            'sched:sched_switch': True,
            'sched:sched_stat_runtime': True,
            'sched:sched_wakeup': True,
            'instructions:u': True,
            'instructions:k': True,#
            'dTLB-load-misses': True,
            'dTLB-loads': True,
            'dTLB-store-misses': True,
            'dTLB-stores': True,
            'iTLB-load-misses': True,
            'iTLB-loads': True,
            'L1-dcache-load-misses': True,
            'L1-dcache-loads': True,
            'L1-dcache-stores': True,
            'L1-icache-load-misses': True,
            'LLC-load-misses': True,
            'LLC-loads': True,
            'LLC-store-misses': True,
            'LLC-stores': True
        }
        for dirpath, dirnames, files in os.walk(path):
            for filename in files:
                data_list = []
                if filename.endswith('.perf'):
                    data_array = pd.read_csv(os.path.join(dirpath, filename), skiprows=1, header=None).values
                    for items in data_array:
                        if feature_vector_dict[items[2]]:
                            data_list.append(items[0])
                            feature_list.append(items[2])
                    #print(data_list)
                    data_path_list[int(filename.split('_')[0])] = (data_list)
                    # data_path_list.append(data_list)
        data_array = np.array(data_path_list)
        return data_array, feature_list

    def read_json_data(self, path, feature):
        data_path_list = [None] * 50
        feature_list = []
        for dirpath, dirnames, files in os.walk(path):
            for filename in files:
                data_list = []
                if filename.endswith('json'):
                    with open(os.path.join(dirpath, filename), 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            elements = line.strip().split("\t")
                            # print(elements)
                            if elements[0] == feature:
                                data_list.append(float(elements[1]))
                                # print(data_list)
                                # data_list.append(float(elements[1]))
                                # print(data_list)
                                # data_list = [items[0] for items in data_list if feature_vector_dict[items[2]]]
                data_path_list[int(filename.split('_')[0])] = data_list
        data_array = np.array(data_path_list)
        #print(data_array.shape)
        #print(data_array.ndim)
        return data_array, list(feature)

    def read_json_data(self, path, feature):
        data_path_list = [None] * 200
        feature_list = []
        for dirpath, dirnames, files in os.walk(path):
            for filename in files:
                data_list = []
                if filename.endswith('json'):
                    with open(os.path.join(dirpath, filename), 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            elements = line.strip().split("\t")
                            # print(elements)
                            if elements[0] == feature:
                                data_list.append(float(elements[1]))
                                # print(data_list)
                                # data_list.append(float(elements[1]))
                                # print(data_list)
                                # data_list = [items[0] for items in data_list if feature_vector_dict[items[2]]]
                data_path_list[int(filename.split('_')[0])] = data_list
        data_array = np.array(data_path_list)
        #print(data_array.shape)
        #print(data_array.ndim)
        return data_array, list(feature)

    def read_net_logs(self, path):
        data_path_list = [0] * 200
        for dirpath, dirname, files in os.walk(path):
            for filename in files:
                feature_list = []
                tcp_connections = {}
                if filename.endswith('.tcp'):
                    with open(os.path.join(dirpath, filename), 'r') as f:
                        lines = f.readlines()[:-1]
                        for line in lines:
                            if line.strip().split()[1] == '[::ffff:130.245.145.210]:80':
                                conn_id = line.strip().split()[2].split(']')[1][1:]
                                snd_cwnd = line.strip().split()[6]
                                #print(snd_cwnd)
                                try:
                                    tcp_connections[conn_id].append(snd_cwnd)
                                except KeyError:
                                    tcp_connections[conn_id] = []
                                    tcp_connections[conn_id].append(snd_cwnd)
                        for connections, values in tcp_connections.items():
                            values = [float(x) for x in values]
                            cwnd_avg = sum(values) / float(len(values))
                            feature_list.append(cwnd_avg)
                        all_cwnd_avg = (round(sum(feature_list)/float(len(feature_list))))
                    data_path_list[int(filename.split('_')[0])] = all_cwnd_avg
        data_array = np.array(data_path_list).reshape(200,1)
        #print(data_array.shape)
        #print(data_array.ndim)
        return data_array



"""def main():
    np.set_printoptions(suppress=True)
    myread = ReadFiles('json')
    myread.read_net_logs('/Users/jnejati/PycharmProjects/PLTSpeed/data/desktop_average_wifi_alexa_orig_inlined/net_logs')
    #myread.read_json_data('/Users/jnejati/PycharmProjects/PLTSpeed/data/desktop_average_wifi_alexa_orig_original/temp_files/wprof_300_5_pro_1', 'load:')

if __name__ == "__main__":
    main()"""
