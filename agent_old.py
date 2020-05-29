# -*- coding: utf-8 -*-
# @File: agent.py
# @Author: Xiaocheng Tang
# @Date:   2020-03-17 17:03:34

import os
import numpy as np
import pandas as pd
import json

zero_threshold = 1e-8

class KMNode(object):
    def __init__(self, id, exception=0, match=None, visit=False):
        self.id, self.exception = id, exception
        self.match, self.visit = match, visit

class KuhnMunkres(object):
    def __init__(self):
        self.matrix = None
        self.x_nodes, self.y_nodes = [], []
        self.minz = float('inf')
        self.x_length, self.y_length = 0, 0
        self.index_x, self.index_y = 0, 1

    def set_matrix(self, x_y_values):
        xs = set([x for x, y, value in x_y_values])
        ys = set([y for x, y, value in x_y_values])
        if len(xs) > len(ys):
            self.index_x = 1
            self.index_y = 0
            xs, ys = ys, xs
        x_dic = {x: i for i, x in enumerate(xs)}
        y_dic = {y: j for j, y in enumerate(ys)}
        self.x_nodes = [KMNode(x) for x in xs]
        self.y_nodes = [KMNode(y) for y in ys]
        self.x_length, self.y_length = len(xs), len(ys)
        self.matrix = np.zeros((self.x_length, self.y_length))
        for row in x_y_values:
            x, y, value = row[self.index_x], row[self.index_y], row[2]
            x_index, y_index = x_dic[x], y_dic[y]
            self.matrix[x_index, y_index] = value
        for i in range(self.x_length):
            self.x_nodes[i].exception = max(self.matrix[i, :])

    def km(self):
        for i in range(self.x_length):
            while True:
                self.minz = float('inf')
                self.set_false(self.x_nodes)
                self.set_false(self.y_nodes)
                if self.dfs(i):
                    break
                self.change_exception(self.x_nodes, -self.minz)
                self.change_exception(self.y_nodes, self.minz)

    def dfs(self, i):
        match_list = []
        while True:
            x_node = self.x_nodes[i]
            x_node.visit = True
            for j in range(self.y_length):
                y_node = self.y_nodes[j]
                if not y_node.visit:
                    t = x_node.exception + y_node.exception - self.matrix[i][j]
                    if abs(t) < zero_threshold:
                        y_node.visit = True
                        match_list.append((i, j))
                        if y_node.match is None:
                            self.set_match_list(match_list)
                            return True
                        else:
                            i = y_node.match
                            break
                    else:
                        if t >= zero_threshold:
                            self.minz = min(self.minz, t)
            else:
                return False
    
    def set_match_list(self, match_list):
        for i, j in match_list:
            x_node, y_node = self.x_nodes[i], self.y_nodes[j]
            x_node.match, y_node.match = j, i
    def set_false(self, nodes):
        for node in nodes:
            node.visit = False
    def change_exception(self, nodes, change):
        for node in nodes:
            if node.visit:
                node.exception += change

    def get_connect_result(self):
        ret = []
        for i in range(self.x_length):
            x_node = self.x_nodes[i]
            j = x_node.match
            y_node = self.y_nodes[j]
            x_id, y_id = x_node.id, y_node.id
            value = self.matrix[i][j]
            if self.index_x == 1 and self.index_y == 0:
                x_id, y_id = y_id, x_id
            ret.append((x_id, y_id, value))
        return ret

    def get_max_value_result(self):
        ret = 0
        for i in range(self.x_length):
            j = self.x_nodes[i].match
            ret += self.matrix[i][j]
        return ret

def run_kuhn_munkres(xys):
    s_xys = set([(x,y) for x,y,t in xys])
    process = KuhnMunkres()
    process.set_matrix(xys)
    process.km()
    res = process.get_connect_result()
    res = [(x,y,t) for x,y,t in res if (x,y) in s_xys]
    return res
def run_kuhn_munkres_min(xys):
    val_max = max([t for x, y, t in xys]) + 1
    xys = [(x, y, val_max - t) for x, y, t in xys]
    res = run_kuhn_munkres(xys)
    res = [(x, y, val_max - t) for x, y, t in res]
    return res

gm = 0.9
n_itv = 24 * 6
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PATH_VALUE = os.path.join(DATA_DIR, 'Vvalue.json')
PATH_HEXS = os.path.join(DATA_DIR, 'hexagon_grid_table.csv')
class Agent(object):
    """ Agent for dispatching and reposition """

    def __init__(self):
        """ Load your trained model and initialize the parameters """
        self.t_start = None
        self._load_hexs()
        self._load_value()

    def _load_value(self):
        with open(PATH_VALUE) as f:
            self.vmat = json.load(f)
            self.vmat = np.array(self.vmat)
            print("vValue", self.vmat.shape)

    def _load_hexs(self):
        from sklearn.neighbors import KDTree
        hexs = np.array(pd.read_csv(PATH_HEXS, usecols=range(1,13), header=None))
        hexs = hexs.reshape((-1,6,2))
        print("hexs tree", hexs.shape)
        hexs_c = np.mean(hexs, axis=1)
        self.tree = KDTree(hexs_c, leaf_size=4)

    def ts2itv(self, ts):
        return min((int(ts)-self.t_start)//600, n_itv - 1)
    def coor2hex(self, coor):
        return self.tree.query([coor])[1][0][0]
    def cal_cost(self, ob):
        t_cur, t_dst = self.ts2itv(ob['timestamp']), self.ts2itv(ob['order_finish_timestamp'] + ob['pick_up_eta'])
        dur = max(t_dst - t_cur, 1)
        rw_u = ob['reward_units'] / dur
        rw = sum([rw_u * gm**i for i in range(dur)])
        h_cur, h_dst = self.coor2hex(ob['driver_location']), self.coor2hex(ob['order_finish_location'])
        adv = self.vmat[h_dst][t_dst] * gm**dur + rw - self.vmat[h_cur][t_cur]
        return adv
    
    def dispatch(self, dispatch_observ):
        """ Compute the assignment between drivers and passengers at each time step
        :param dispatch_observ: a list of dict, the key in the dict includes:
            order_id, int
            driver_id, int
            order_driver_distance, float
            order_start_location, a list as [lng, lat], float
            order_finish_location, a list as [lng, lat], float
            driver_,location, a list as [lng, lat], float
            timestamp, int
            order_finish_timestamp, int
            day_of_week, int
            reward_units, float
            pick_up_eta, float

        :return: a list of dict, the key in the dict includes:
            order_id and driver_id, the pair indicating the assignment
        """
        if(self.t_start is None):# set the start time to the correct 16:00
            first_t = pd.to_datetime(dispatch_observ[0]['timestamp'], unit='s')
            if(first_t.hour < 16):
                ts = first_t - pd.Timedelta('1d')
            else:
                ts = first_t
            ts = pd.Timestamp("%d-%d-%d 16:00:00"%(ts.year, ts.month, ts.day))
            self.t_start = int(ts.timestamp())
            print("start time", first_t, ts, self.t_start)
        
        # res = self.dispatch_greedy(dispatch_observ)
        # res = self.dispatch_min_dist(dispatch_observ)
        res = self.dispatch_max_val(dispatch_observ)

        dispatch_action = [dict(order_id=xy[0], driver_id=xy[1]) for xy in res]
        return dispatch_action

    def dispatch_max_val(self, dispatch_observ):
        xys = [(x['order_id'], x['driver_id'], self.cal_cost(x)) for x in dispatch_observ]
        # print(xys)
        res = run_kuhn_munkres(xys)
        return res

    def dispatch_min_dist(self, dispatch_observ):
        xys = [(x['order_id'], x['driver_id'], x['pick_up_eta']) for x in dispatch_observ]
        res = run_kuhn_munkres_min(xys)
        return res

    def dispatch_greedy(self, dispatch_observ):
        dispatch_observ.sort(key=lambda od_info: od_info['reward_units'], reverse=True)
        assigned_order, assigned_driver = set(), set()
        xys = []
        for od in dispatch_observ:
            # make sure each order is assigned to one driver, and each driver is assigned with one order
            if (od["order_id"] in assigned_order) or (od["driver_id"] in assigned_driver):
                continue
            assigned_order.add(od["order_id"])
            assigned_driver.add(od["driver_id"])
            xys.append((od["order_id"], od["driver_id"], od['reward_units']))
        return xys

    def reposition(self, repo_observ):
        """ Compute the reposition action for the given drivers
        :param repo_observ: a dict, the key in the dict includes:
            timestamp: int
            driver_info: a list of dict, the key in the dict includes:
                driver_id: driver_id of the idle driver in the treatment group, int
                grid_id: id of the grid the driver is located at, str
            day_of_week: int

        :return: a list of dict, the key in the dict includes:
            driver_id: corresponding to the driver_id in the od_list
            destination: id of the grid the driver is repositioned to, str
        """
        repo_action = []
        for driver in repo_observ['driver_info']:
            # the default reposition is to let drivers stay where they are
            repo_action.append({'driver_id': driver['driver_id'], 'destination': driver['grid_id']})
        return repo_action
