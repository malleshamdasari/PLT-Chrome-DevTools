import collections
import networkx as nx
import json
#import matplotlib.pyplot as plt
import os
import logging
import shutil
import tldextract
import time


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# create console handler and set level to info
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# create error file handler and set level to error
handler = logging.FileHandler(os.path.join('./', "error.log"), "w", encoding=None, delay="true")
handler.setLevel(logging.ERROR)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# create debug file handler and set level to debug
handler = logging.FileHandler(os.path.join('./', "all.log"), "w")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class JsonToDag():
    dep_visited_nodes = set()

    # path_orig_temp = './data_mobile_s4_unlimited/temp_files'
    def __init__(self, my_file):
        self.my_file = my_file
        self.json_orig = open(self.my_file)
        self.j_orig = json.load(self.json_orig)

    def return_attr(self, object_id, attr, my_type=None):
        attr_dict = {'receivedTime': 'e_time', 's_time': 's_time', 'type': 'type', 'len': 'len', 'url': 'url'}
        if attr == 'url':
            if object_id.startswith('download'):
                for my_index in range(len(self.j_orig['objs'])):
                    try:
                        if self.j_orig['objs'][my_index]['download']['id'] == object_id:
                            attr_value = self.j_orig['objs'][my_index]['url']
                    except KeyError:
                        continue
            else:
                for j_index in range(len(self.j_orig['objs'])):
                    for k_index in range(len(self.j_orig['objs'][j_index]['comps'])):
                        if self.j_orig['objs'][j_index]['comps'][k_index]['id'] == object_id:
                            attr_value = self.j_orig['objs'][j_index]['url']
            return attr_value

        elif object_id.startswith('download'):
            for my_index in range(len(self.j_orig['objs'])):
                try:
                    if self.j_orig['objs'][my_index]['download']['id'] == object_id:
                        attr_value = self.j_orig['objs'][my_index]['download'][attr]
                except KeyError:
                    continue

        else:
            for j_index in range(len(self.j_orig['objs'])):
                for k_index in range(len(self.j_orig['objs'][j_index]['comps'])):
                    if self.j_orig['objs'][j_index]['comps'][k_index]['id'] == object_id:
                        attr_value = self.j_orig['objs'][j_index]['comps'][k_index][attr_dict[attr]]

        if my_type == 'string': return attr_value
        return float(attr_value)

    def get_max_end_time(self, my_list):
        my_max = 0
        longest_node = ''
        max_end = 0
        for my_node in my_list:
            max_end = self.return_attr(my_node, 'receivedTime')
            if max_end > my_max:
                my_max = max_end
                longest_node = my_node
        return longest_node, max_end

    def get_all_nodes(self):
        all_nodes = []
        if len(self.j_orig['objs']) > 1:
            for i in range(len(self.j_orig['objs'])):
                try:
                    all_nodes.append(self.j_orig['objs'][i]['download']['id'])
                except KeyError:
                    continue
            for j in range(len(self.j_orig['objs'])):
                for k in range(len(self.j_orig['objs'][j]['comps'])):
                    try:
                        all_nodes.append(self.j_orig['objs'][j]['comps'][k]['id'])
                    except  KeyError:
                        continue
            return all_nodes
        else:
            return None

    def populate_attributes(self, object_id):
        if object_id.startswith('download'):
            for i in range(len(self.j_orig['objs'])):
                try:
                    if self.j_orig['objs'][i]['download']['id'] == object_id:
                        self.G.node[object_id]['receiveFirst'] = self.j_orig['objs'][i]['download']['receiveFirst']
                        self.G.node[object_id]['len'] = self.j_orig['objs'][i]['download']['len']
                        self.G.node[object_id]['dns'] = self.j_orig['objs'][i]['download']['dns']
                        self.G.node[object_id]['dnsEnd'] = self.j_orig['objs'][i]['download']['dnsEnd']
                        self.G.node[object_id]['receiveHeadersEnd'] = self.j_orig['objs'][i]['download'][
                            'receiveHeadersEnd']
                        self.G.node[object_id]['sslEnd'] = self.j_orig['objs'][i]['download']['sslEnd']
                        self.G.node[object_id]['connectEnd'] = self.j_orig['objs'][i]['download']['connectEnd']
                        self.G.node[object_id]['connectStart'] = self.j_orig['objs'][i]['download']['connectStart']
                        self.G.node[object_id]['id'] = self.j_orig['objs'][i]['download']['id']
                        self.G.node[object_id]['receivedTime'] = float(
                            self.j_orig['objs'][i]['download']['receivedTime'])
                        self.G.node[object_id]['sslStart'] = self.j_orig['objs'][i]['download']['sslStart']
                        self.G.node[object_id]['dnsStart'] = self.j_orig['objs'][i]['download']['dnsStart']
                        self.G.node[object_id]['receiveLast'] = self.j_orig['objs'][i]['download']['receiveLast']
                        self.G.node[object_id]['s_time'] = self.j_orig['objs'][i]['download']['s_time']
                        self.G.node[object_id]['dnsStart'] = self.j_orig['objs'][i]['download']['dnsStart']
                        self.G.node[object_id]['sendStart'] = self.j_orig['objs'][i]['download']['sendStart']
                        self.G.node[object_id]['sendEnd'] = self.j_orig['objs'][i]['download']['sendEnd']
                        self.G.node[object_id]['type'] = self.j_orig['objs'][i]['download']['type']
                        self.G.node[object_id]['when_comp_start'] = self.j_orig['objs'][i]['when_comp_start']
                        self.G.node[object_id]['url'] = self.j_orig['objs'][i]['url']
                        self.G.node[object_id]['parent_id'] = self.j_orig['objs'][i]['id']
                except KeyError:
                    continue

        else:
            for j in range(len(self.j_orig['objs'])):
                for k in range(len(self.j_orig['objs'][j]['comps'])):
                    try:
                        if self.j_orig['objs'][j]['comps'][k]['id'] == object_id:
                            self.G.node[object_id]['s_time'] = self.j_orig['objs'][j]['comps'][k]['s_time']
                            self.G.node[object_id]['time'] = self.j_orig['objs'][j]['comps'][k]['time']
                            self.G.node[object_id]['type'] = self.j_orig['objs'][j]['comps'][k]['type']
                            self.G.node[object_id]['id'] = self.j_orig['objs'][j]['comps'][k]['id']
                            self.G.node[object_id]['e_time'] = self.j_orig['objs'][j]['comps'][k]['e_time']
                            self.G.node[object_id]['when_comp_start'] = self.j_orig['objs'][j]['when_comp_start']
                            self.G.node[object_id]['url'] = self.j_orig['objs'][j]['url']
                            self.G.node[object_id]['parent_id'] = self.j_orig['objs'][j]['id']
                    except KeyError:
                        continue

    def add_nodes(self):
        if self.get_all_nodes():
            all_nodes_list = self.get_all_nodes()
        else:
            return None
        self.G = nx.DiGraph(n_download_no_trigger=self.j_orig['n_download_no_trigger'],
                            start_activity=self.j_orig['start_activity'],
                            name=self.j_orig['name'],
                            load_activity=self.j_orig['load_activity'])

        self.G.add_node(self.G.graph['start_activity'])
        self.populate_attributes(self.G.graph['start_activity'])
        self.G.add_node(self.j_orig['objs'][0]['comps'][0]['id'])
        self.populate_attributes(self.j_orig['objs'][0]['comps'][0]['id'])
        # Edge weight means difference between next edge s_time and current edge s_time
        self.G.add_edge(self.G.graph['start_activity'], self.j_orig['objs'][0]['comps'][0]['id'],
                        weight=self.j_orig['objs'][0]['comps'][0]['s_time'] - float(
                            self.j_orig['objs'][0]['download']['s_time']), id='dep0')
        base_nodes = ['download_0', 'r1_c1']
        remaining_list = list(set(all_nodes_list) - set(base_nodes))
        for my_node in remaining_list:
            self.G.add_node(my_node)
            self.populate_attributes(my_node)
        return True

    def add_dependencies(self, object_id=None):
        time_diff = 0
        # Assumption: never there is a download as a1.
        javascript_type_list = ['application/x-javascript', 'application/javascript', 'application/ecmascript',
                                'text/javascript', 'text/ecmascript', 'application/json', 'javascript/text', 1, 2, 3]
        css_type_list = ['text/css', 'css/text', 4]

        ###
        #  Sequential comps deps (r1_c1->r1_c2->r1_c3,...)
        ###

        dep_dict = dict()
        for i in range(len(self.j_orig['objs'])):
            if len(self.j_orig['objs'][i]['comps']) > 1:
                try:
                    dep_dict[self.j_orig['objs'][i]['id']] = self.j_orig['objs'][i]['comps']
                except KeyError:
                    continue
        all_comps = {}
        for key, dep_list in dep_dict.items():
            for my_index, my_dict in enumerate(dep_list):
                try:
                    all_comps[key].append(my_dict['id'])
                except KeyError:
                    all_comps[key] = []
                    all_comps[key].append(my_dict['id'])

        for index, value in all_comps.items():
            # print(index, value)
            tobe_removed = []
            for my_index, seq_comp in enumerate(value):
                if self.return_attr(seq_comp, 's_time') < 0 or self.return_attr(seq_comp, 'receivedTime') < 0:
                    tobe_removed.append(seq_comp)
            sequential_eval_html = [x for x in value if x not in tobe_removed]
            # print('sequential_eval_html', sequential_eval_html)
            for my_index, seq_com in enumerate(sequential_eval_html):
                a1_s_time = self.return_attr(sequential_eval_html[my_index], 's_time')
                if my_index == len(sequential_eval_html) - 1:
                    break
                a2_s_time = self.return_attr(sequential_eval_html[my_index + 1], 's_time')
                if not self.G.has_edge(sequential_eval_html[my_index], sequential_eval_html[my_index + 1]):
                    # print("Adding Edges: ", sequential_eval_html[my_index], sequential_eval_html[my_index + 1])
                    self.G.add_edge(sequential_eval_html[my_index], sequential_eval_html[my_index + 1],
                                    weight=a2_s_time - a1_s_time,
                                    id='seq_comps_dep' + str(my_index))

        ###
        #  F2 dependecncy between download js/css then evaljs/evalcss
        ###
        for j_index in range(len(self.j_orig['objs'])):
            for k_index in range(len(self.j_orig['objs'][j_index]['comps'])):
                try:
                    if self.j_orig['objs'][j_index]['comps'][k_index]['type'] in javascript_type_list or \
                                    self.j_orig['objs'][j_index]['comps'][k_index]['type'] in css_type_list or \
                                    self.j_orig['objs'][j_index]['download']['type'] in javascript_type_list or \
                                    self.j_orig['objs'][j_index]['download']['type'] in css_type_list:
                        my_a1_s_time = float(self.j_orig['objs'][j_index]['download']['s_time'])
                        my_a2_s_time = float(self.j_orig['objs'][j_index]['comps'][k_index]['s_time'])
                        self.G.add_edge(self.j_orig['objs'][j_index]['download']['id'],
                                        self.j_orig['objs'][j_index]['comps'][k_index]['id'],
                                        weight=my_a2_s_time - my_a1_s_time, id='F2dep' + str(j_index))
                    elif self.j_orig['objs'][j_index]['url'].endswith('.js') or \
                            self.j_orig['objs'][j_index]['url'].endswith('.css'):
                        my_a1_s_time = float(self.j_orig['objs'][j_index]['download']['s_time'])
                        my_a2_s_time = float(self.j_orig['objs'][j_index]['comps'][k_index]['s_time'])
                        self.G.add_edge(self.j_orig['objs'][j_index]['download']['id'],
                                        self.j_orig['objs'][j_index]['comps'][k_index]['id'],
                                        weight=my_a2_s_time - my_a1_s_time, id='F2dep' + str(j_index))
                except KeyError:
                    continue
        ###
        # add dependencies from deps[]
        ###
        all_nodes_reference = self.get_all_nodes()
        if object_id is None:
            fringe_set = set()
            for j in range(len(self.j_orig['deps'])):
                if not self.j_orig['deps'][j]['a1'] in all_nodes_reference:
                    logging.warning(self.j_orig['deps'][j]['a1'] + ' is in deps but not in objs! Object skipped')
                    continue
                if not self.j_orig['deps'][j]['a2'] in all_nodes_reference:
                    logging.warning(self.j_orig['deps'][j]['a2'] + ' is in deps but not in objs! Object skipped')
                    continue
                fringe_set.add(self.j_orig['deps'][j]['a1'])
            while fringe_set:
                my_object_id = fringe_set.pop()
                for i in range(len(self.j_orig['deps'])):
                    if self.j_orig['deps'][i]['a1'] == my_object_id and not self.j_orig['deps'][i][
                        'a1'] in self.dep_visited_nodes:
                        a2_s_time = self.return_attr(self.j_orig['deps'][i]['a2'], 's_time')
                        if not self.G.has_edge(my_object_id, self.j_orig['deps'][i]['a2']):
                            self.G.add_edge(my_object_id, self.j_orig['deps'][i]['a2'],
                                            weight=a2_s_time - float(self.G.node[my_object_id]['s_time']),
                                            id=self.j_orig['deps'][i]['id'])
                self.dep_visited_nodes.add(object_id)
        else:
            if object_id not in self.dep_visited_nodes:
                for i in range(len(self.j_orig['deps'])):
                    if self.j_orig['deps'][i]['a1'] == object_id:
                        a2_s_time = self.return_attr(self.j_orig['deps'][i]['a2'], 's_time')
                        if not self.G.has_edge(object_id, self.j_orig['deps'][i]['a2']):
                            self.G.add_edge(object_id, self.j_orig['deps'][i]['a2'],
                                            weight=a2_s_time - float(self.G.node[object_id]['s_time']),
                                            id=self.j_orig['deps'][i]['id'])

    def h_recur(self, G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5,
                pos=None, parent=None, parsed=[]):
        if (root not in parsed):
            parsed.append(root)
            if pos == None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            neighbors = G.neighbors(root)
            if parent != None and parent in neighbors:
                neighbors.remove(parent)
            if len(neighbors) != 0:
                dx = width / len(neighbors)
                nextx = xcenter - width / 2 - dx / 2
                for neighbor in neighbors:
                    nextx += dx
                    pos = self.h_recur(G, neighbor, width=dx, vert_gap=vert_gap,
                                       vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos,
                                       parent=root, parsed=parsed)
        return pos

    def draw_graph(self, my_list=None):
        # pos = nx.spring_layout(self.G, iterations=20)
        pos = nx.graphviz_layout(self.G, prog='sfdp', root='downlod_0', args="-Grankdir=LR")
        # pos = nx.spring_layout(self.G,fixed = ['r1_c1'], scale=2)
        # print(pos)
        # pos = self.h_recur(self.G, 'download_0')
        # nx.draw(self.G, pos=pos, with_labels=True)
        for node in self.G.nodes():
            self.G.node[node]['category'] = 'type_A'
        if my_list:
            for my_nodes2 in my_list:
                self.G.node[my_nodes2]['category'] = 'critical_path'
        color_map = {'type_A': 'y', 'critical_path': '#FFC266'}
        ## construct a list of colors then pass to node_color
        if my_list:
            nx.draw(self.G, pos=pos, node_size=200,
                    node_color=[color_map[self.G.node[node]['category']] for node in self.G])
        else:
            nx.draw(self.G, pos=pos, node_size=200, node_color='y')

        node_labels = nx.get_node_attributes(self.G, 'id')
        nx.draw_networkx_labels(self.G, pos, labels=node_labels, font_size=8, font_color='r', font_weight='bold',
                                font_family='sans-serif')
        """edge_labels = nx.get_edge_attributes(self.G, 'weight')
        for key, value in edge_labels.items():
            edge_labels[key] = round(value, 4)"""
        # print('edge lbl', edge_labels)
        # nx.draw_networkx_edge_labels(self.G, pos, labels=edge_labels)
        plt.show()
        # plt.savefig("origin.png")

    def get_last_node(self, graph, exclude_list=None):
        last_node = 'download_0'
        for n, d in graph.nodes_iter(data=True):
            if exclude_list:
                if not n in exclude_list and self.return_attr(n, 'receivedTime') > self.return_attr(last_node,
                                                                                                    'receivedTime'):
                    last_node = n
            else:
                if self.return_attr(n, 'receivedTime') > self.return_attr(last_node, 'receivedTime'):
                    last_node = n
        return last_node

    def get_last_node_target(self, graph, target):
        last_node = 'download_0'
        for n, d in graph.nodes_iter(data=True):
            if self.return_attr(n, 'receivedTime') > self.return_attr(last_node, 'receivedTime'):
                if self.return_attr(n, 'receivedTime') < self.return_attr(target, 'receivedTime'):
                    last_node = n
        return last_node

    def get_first_node(self, graph):
        first_node = self.get_last_node(graph)
        for n, d in graph.nodes_iter(data=True):
            if self.return_attr(n, 's_time') < self.return_attr(first_node, 's_time'):
                first_node = n
        return first_node

    def find_cp(self, cur_node, cr_list):
        ###
        # New critical path function
        ###
        cr_list.append(cur_node)
        if cur_node == 'download_0':
            return
        pred = self.G.predecessors(cur_node)
        if len(pred) == 0:
            new_last_node = self.get_last_node_target(self.G, cur_node)
            self.find_cp(new_last_node, cr_list)
        else:
            new_last_node = 'download_0'
            for n in pred:
                if self.return_attr(n, 'receivedTime') > self.return_attr(new_last_node, 'receivedTime'):
                    new_last_node = n
            self.find_cp(new_last_node, cr_list)
        return cr_list

    def findCriticalPath(self):
        num_objs_cp = 0
        num_objs_all = 0
        num_bytes_all = 0
        num_bytes_cp = 0
        download_dns = 0
        domains_cp = set()
        domains_all = set()
        num_js_cp = 0
        num_js_all = 0
        num_css_cp = 0
        num_css_all = 0
        stats_dic = collections.OrderedDict([
            ('load', 0),
            ('HTMLParse', 0.0),
            ('TTFB', 0.0),
            ('Parse', 0.0),
            ('PostParse', 0.0),
            ('level', 0),
            ('time_download', 0.0),
            ('time_comp', 0.0),
            ('time_block', 0.0),
            ('whatif_matrix', 0.0),
            ('download_blocking', 0.0),
            ('download_proxy', 0.0),
            ('download_dns', 0.0),
            ('download_conn', 0.0),
            ('download_ssl', 0.0),
            ('download_send:', 0.0),
            ('download_receiveFirst', 0.0),
            ('download_receiveLast', 0.0),
            ('parse_style', 0.0),
            ('parse_script', 0.0),
            ('parse_layout', 0.0),
            ('parse_paint', 0.0),
            ('parse_other', 0.0),
            ('parse_undefined', 0.0),
            ('dep_D2E', 0.0),
            ('dep_E2D_html', 0.0),
            ('dep_E2D_css', 0.0),
            ('dep_E2D_js', 0.0),
            ('dep_E2D_timer', 0.0),
            ('dep_RFB', 0.0),
            ('dep_HOL_css', 0.0),
            ('dep_HOL_js', 0.0),
            ('time_download_html', 0.0),
            ('time_download_css', 0.0),
            ('time_download_js', 0.0),
            ('time_download_img', 0.0),
            ('time_download_o', 0.0),
            ('time_block_css', 0.0),
            ('time_block_js', 0.0),
            ('time_ttfb', 0.0),
            ('num_domains_cp', 0.0),
            ('num_domains_all', 0),
            ('text_domains_cp', {}),
            ('text_domains_all', {}),
            ('num_bytes_cp', 0.0),
            ('num_bytes_all', 0.0),
            ('num_send_cp', 0.0),
            ('num_send_all', 0.0),
            ('num_conn_cp', 0.0),
            ('num_conn_all', 0.0),
            ('num_objs_cp', 0.0),
            ('num_objs_all', 0.0),
            ('num_js_cp', 0.0),
            ('num_js_all', 0.0),
            ('num_css_cp', 0.0),
            ('num_css_all', 0.0),
            ('text_domain_tcp_net_cp', []),
            ('text_domain_tcp_net_all', []),
            ('critical_path', [])
        ])

        javascript_type_list = ['application/x-javascript', 'application/javascript', 'application/ecmascript',
                                'text/javascript', 'text/ecmascript', 'application/json', 'javascript/text']
        css_type_list = ['text/css', 'css/text']
        time_gap = 0
        SUBTRACT_TIME_GAP = 0
        last_node = self.get_last_node(self.G)
        crit_path_list = []
        crit_path_list = self.find_cp(last_node, crit_path_list)
        crit_path_list.reverse()
        for cp_index, current_node in enumerate(crit_path_list):
            info = self.return_attr(current_node, 'type', 'string')
            info_url = self.return_attr(current_node, 'url')
            cur_recevied_Time = self.return_attr(current_node, 'receivedTime')
            cur_s_time = self.return_attr(current_node, 's_time')
            if cp_index < len(crit_path_list) - 1:
                next_s_time = self.return_attr(crit_path_list[cp_index + 1], 's_time')
                time_gap = next_s_time - cur_recevied_Time
                if time_gap >= 0:
                    stats_dic['time_block'] = stats_dic['time_block'] + time_gap
                else:
                    SUBTRACT_TIME_GAP = 1
            else:
                SUBTRACT_TIME_GAP = 0

            if cur_recevied_Time >= 0 and cur_s_time >= 0:
                if current_node.startswith('download'):
                    num_bytes_cp = num_bytes_cp + self.return_attr(current_node, 'len')
                    download_dns = download_dns + self.G.node[current_node]['dns']
                    domain_name_tuple = tldextract.extract(self.G.node[current_node]['url'])
                    domain_name = domain_name_tuple.domain + "." + domain_name_tuple.suffix
                    domains_cp.add(domain_name)

                    if info.split("/")[1] == 'html':
                        stats_dic['time_download_html'] = stats_dic[
                                                              'time_download_html'] + cur_recevied_Time - cur_s_time + (
                                                              SUBTRACT_TIME_GAP * time_gap)
                    elif info.split("/")[0] == 'image':
                        stats_dic['time_download_img'] = stats_dic[
                                                             'time_download_img'] + cur_recevied_Time - cur_s_time + (
                                                             SUBTRACT_TIME_GAP * time_gap)
                    elif info in css_type_list or info_url in css_type_list:
                        stats_dic['time_download_css'] = stats_dic[
                                                             'time_download_css'] + cur_recevied_Time - cur_s_time + (
                                                             SUBTRACT_TIME_GAP * time_gap)
                        num_css_cp = num_css_cp + 1
                    elif info in javascript_type_list or info_url in javascript_type_list:
                        stats_dic['time_download_js'] = stats_dic[
                                                            'time_download_js'] + cur_recevied_Time - cur_s_time + (
                                                            SUBTRACT_TIME_GAP * time_gap)
                        num_js_cp = num_js_cp + 1
                    else:
                        stats_dic['time_download_o'] = stats_dic['time_download_o'] + cur_recevied_Time - cur_s_time + (
                            SUBTRACT_TIME_GAP * time_gap)

                        # adding computation time on critical path
                else:
                    if current_node == 'r1_c1': stats_dic['TTFB'] = self.return_attr(current_node, 's_time')
                    if info == 'evalhtml':
                        stats_dic['HTMLParse'] = stats_dic['HTMLParse'] + (cur_recevied_Time - cur_s_time) + (
                            SUBTRACT_TIME_GAP * time_gap)
                    elif info == 1 or info == 2 or info == 3:  # evaljs
                        stats_dic['parse_script'] = stats_dic['parse_script'] + (cur_recevied_Time - cur_s_time) + (
                            SUBTRACT_TIME_GAP * time_gap)
                    elif info == 4:  # evalcss
                        stats_dic['parse_style'] = stats_dic['parse_style'] + (cur_recevied_Time - cur_s_time) + (
                            SUBTRACT_TIME_GAP * time_gap)
                    else:
                        stats_dic['parse_undefined'] = stats_dic['parse_undefined'] + (
                            cur_recevied_Time - cur_s_time) + (SUBTRACT_TIME_GAP * time_gap)

                # time_gap = next_s_time - cur_recevied_Time
                SUBTRACT_TIME_GAP = 0
                """if time_gap >= 0:
                    stats_dic['time_block'] = stats_dic['time_block'] + time_gap
                else:
                    SUBTRACT_TIME_GAP = 1"""

            else:
                pass
                # return None

        if crit_path_list:
            stats_dic['time_download'] = stats_dic['time_download_html'] + stats_dic['time_download_img'] + \
                                         stats_dic['time_download_css'] + stats_dic['time_download_js'] + \
                                         stats_dic['time_download_o']
            stats_dic['time_comp'] = stats_dic['parse_undefined'] + stats_dic['parse_style'] + \
                                     stats_dic['parse_script'] + stats_dic['HTMLParse'] + stats_dic['time_block']

            # stats_dic['load'] = stats_dic['time_download'] + stats_dic['time_comp'

            stats_dic['load'] = self.return_attr(self.get_last_node(self.G), 'receivedTime')

            stats_dic['num_objs_cp'] = str(len(crit_path_list))
            stats_dic['num_objs_all'] = len(self.G.node)
            for my_node in self.G.node:
                if my_node.startswith('download'):
                    num_bytes_all = num_bytes_all + self.G.node[my_node]['len']
                    if self.G.node[my_node]['type'] in javascript_type_list:
                        num_js_all = num_js_all + 1
                    if self.G.node[my_node]['type'] in css_type_list:
                        num_css_all = num_css_all + 1

                    domain_name_tuple = tldextract.extract(self.G.node[my_node]['url'])
                    domain_name = domain_name_tuple.domain + "." + domain_name_tuple.suffix
                    domains_all.add(domain_name)

            stats_dic['num_bytes_cp'] = num_bytes_cp
            stats_dic['num_bytes_all'] = num_bytes_all
            stats_dic['num_js_cp'] = num_js_cp
            stats_dic['num_js_all'] = num_js_all
            stats_dic['num_css_cp'] = num_css_cp
            stats_dic['num_css_all'] = num_css_all
            stats_dic['num_css_all'] = num_css_all
            stats_dic['download_dns'] = download_dns
            stats_dic['domains_all'] = len(domains_all)
            stats_dic['domains_cp'] = len(domains_cp)
            stats_dic['critical_path'] = crit_path_list

        return crit_path_list, stats_dic


def json_to_dag(path):
    if os.path.isdir('./temp_files/wprof_300_5_pro_1'):
        for root, dirs, l_files in os.walk('./temp_files/wprof_300_5_pro_1'):
            for f in l_files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
    else:
        os.makedirs('./temp_files/wprof_300_5_pro_1')
    #path = '/Users/jnejati/PycharmProjects/PLTSpeed/data/desktop_average_wifi_original_orig_newperfsched/graphs'
    #path = '/home/jnejati/PLTSpeed/desktop_average_4g_orig_orig_b2d100_www.wikipedia.org/graphs'
    #path = './graphs_t'

    #paths_orig_temp = '/Users/jnejati/PycharmProjects/page_speed/data/mobile_average_wifi_mixed500_orig_inline/temp_files_orig'
    # paths_orig_temp = './graphs_t/temp_files_orig'

    dirs = os.listdir(path)  # return list of files in path
    #dirs_orig_temp = os.listdir(paths_orig_temp)
    i = 0
    bad_count = 0

    for files in dirs:
        if files.endswith('.json'):
            files = os.path.join(path, files)
            logger.info("\n" + str(i) + '- Opening ' + files)
            my_graph = JsonToDag(files)
            if my_graph.add_nodes():
                my_graph.add_dependencies()
                print('load: ', my_graph.return_attr(my_graph.get_last_node(my_graph.G), 'receivedTime'))
                print('last_node: ', my_graph.get_last_node(my_graph.G))
                # my_graph.draw_graph()
                critical_path, stats_result = my_graph.findCriticalPath()
                if critical_path:
                    """for orig_temps in dirs_orig_temp:
                        if orig_temps.startswith('original.testbed.localhost'):
                            if len(orig_temps.split('_')[:-1]) > 1:
                                if orig_temps.split('_')[:-1][1] in files:
                                    file = open(os.path.join(paths_orig_temp, orig_temps), 'r')
                                    load = 0
                                    for line in file:
                                        list1 = line.split(':')
                                        if str(list1[0]) == "load":
                                            load = float(list1[1]) * 1000
                                            temp = stats_result['load']
                                            stats_result['load'] = load
                                            stats_result['PostParse'] = load - temp
                                    print(files)"""
                    logger.info('Analyzing ' + files)
                    fname = files.split("/")[-1].split("_.")[0]
                    file = open('./temp_files/wprof_300_5_pro_1/' + fname, 'w+')
                    if stats_result:
                        for key, value in stats_result.items():
                            file.write(str(key) + ":\t" + str(value) + "\n")
                        logger.info('Critical Path: ')
                        logger.info(critical_path)
                        logger.info('Done!')
                    else:
                        logger.error('Negative time values! skipping json file')
                    file.close()
                else:
                    logger.error(str(i) + '- No critical path: ' + files + ' skipped')
                    bad_count += 1
            else:
                logger.warning(files + ': Not a usable JSON file')

        else:
            logger.warning(files + ': Not a JSON file?')
        i += 1

    logger.info('------------------------------------------------------------------------------------------')
    logger.info('Total: ' + str(i) + '\nSkipped: ' + str(bad_count) + '\nProcessed: ' + str(i - bad_count))
    # my_graph.draw_graph(critical_path)


def main():
    data_folder_list = ['desktop_average_4g_compressed_orig_b0200d400_www.alexa.com',
                    'desktop_average_4g_compressed_orig_b0200d400_www.apple.com',
                    'desktop_average_4g_compressed_orig_b0200d400_www.craigslist.org',
                    'desktop_average_4g_compressed_orig_b0200d400_www.msn.com',
                    'desktop_average_4g_compressed_orig_b0200d400_www.newrepublic.com',
                    'desktop_average_4g_compressed_orig_b0200d400_www.thinkprogress.org',
                    'desktop_average_4g_compressed_orig_b0200d400_www.townhall.com',
                    'desktop_average_4g_compressed_orig_b0200d400_www.wikipedia.org'] 
    root_folder = '/home/jnejati/PLTSpeed'
    for folder in data_folder_list:
        cur_folder = os.path.join(root_folder, folder)
        json_to_dag(os.path.join(cur_folder, 'graphs'))
        shutil.move(os.path.join(root_folder, 'collectors', 'temp_files'), cur_folder)

if __name__ == '__main__':
    main()
