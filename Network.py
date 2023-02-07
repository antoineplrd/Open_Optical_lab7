import copy
import json
import math
from math import *
import random
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from Connection import Connection
from Lightpath import Lightpath
from Node import Node
from Line import Line
from Signal_information import Signal_information
from scipy.special import erfcinv
from statistics import mean


class Network:
    def __init__(self, file):
        self._nodes = dict()
        self._lines = dict()

        open_Json = open(file, "r")
        data = json.loads(open_Json.read())

        for i in data:
            label = i
            connected_nodes = list()
            connected_lines = list()

            for k in data[i]["position"]:
                connected_lines.append(k)
            for j in data[i]["connected_nodes"]:
                # implementation for line between 2 nodes: AB, BA
                connected_nodes.append(j)
            switching_matrix = data[i]["switching_matrix"]
            if "transceiver" in data[i]:
                transceiver = data[i]["transceiver"]
            else:
                transceiver = None

            node = Node(label, (connected_lines[0], connected_lines[1]), connected_nodes,
                        switching_matrix, transceiver)
            self._nodes.update({label: node})
            # print(self._nodes.get(i).switching_matrix) print of the switching matrix for each node

        open_Json.close()

        for i in self._nodes:
            # length between 2 nodes
            labelX = self._nodes.get(i).position[0]
            labelY = self._nodes.get(i).position[1]
            for j in self._nodes.get(i).connected_nodes:
                label_lines = i + j
                nextNodeX = self._nodes.get(j).position[0]
                nextNodeY = self._nodes.get(j).position[1]
                Distance_lines = sqrt((nextNodeX - labelX) ** 2 + (nextNodeY - labelY) ** 2)

                line = Line(label_lines, Distance_lines)
                self._lines.update({label_lines: line})

        self.connect()
        self._weighted_paths = self.ex5()
        self._route_space = self.chanel_availability()

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, lines):
        self._lines = lines

    def connect(self):
        for i in self._lines:
            successive_nodes = dict()
            successive_nodes.update({i[0]: self._nodes.get(i[0])})
            successive_nodes.update({i[1]: self._nodes.get(i[1])})
            self.lines.get(i).successive = successive_nodes
        for i in self._nodes:
            successive_lines = dict()
            for j in self._nodes.get(i).connected_nodes:
                successive_lines.update({i + j: self._lines.get(i + j)})
            self._nodes.get(i).successive = successive_lines

    def find_paths(self, start_node, end_node, path=None):

        if path is None:
            path = []
        graph_dict = self._nodes.get(start_node).connected_nodes
        path = path + [start_node]
        if start_node == end_node:
            return [path]
        paths = []
        for actual_node in graph_dict:
            if actual_node not in path:
                extended_paths = self.find_paths(actual_node, end_node, path)
                for p in extended_paths:
                    paths.append(p)
        return paths

    def propagate(self, signal_information):
        propagate = self._nodes.get(signal_information.path[0]).propagate(signal_information, True)
        self._route_space = self.chanel_availability()  # We update for each new path the avability
        return propagate

    def probe(self, signal_information):
        return self._nodes.get(signal_information.path[0]).probe(signal_information)

    def draw(self):
        G = nx.Graph()
        for i in self._nodes:
            G.add_nodes_from(self._nodes.get(i).label, pos=(self._nodes.get(i).position[0],
                                                            self._nodes.get(i).position[1]))
        for j in self._lines:
            G.add_edges_from([(j[0], j[1])])
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True)
        plt.show()

    def ex5(self):
        var = list()
        path = list()
        for i in self._nodes:
            for j in self._nodes:
                if not (var.__contains__((i, j)) or i == j):
                    var.append((i, j))
                    path.append(self.find_paths(i, j))

        test = list()

        for i in path:
            for j in i:
                path_cpy = j[:]
                path_cpy = '->'.join(path_cpy)
                signal_information = self.probe(Signal_information(0.001, j))
                ratio = 10 * math.log10(signal_information.signal_power / signal_information.noise_power)
                test.append(list([path_cpy, signal_information.latency, signal_information.noise_power, ratio]))

        data = {

            "Paths": [i[0] for i in test],
            "Latency (s)": [i[1] for i in test],
            "Noise power (W)": [i[2] for i in test],
            "Signal/noise (dB)": [i[3] for i in test]
        }
        df = pd.DataFrame(data)

        return df

    def find_best_snr(self, input_node, output_node):
        Dataframe = self._weighted_paths
        Dataframe_occupancy = self._route_space
        all_paths = Dataframe['Paths'].tolist()
        all_noise_radio = Dataframe['Signal/noise (dB)'].tolist()
        all_paths_occupancy = Dataframe_occupancy['Paths'].tolist()
        all_occupancy = Dataframe_occupancy['availability'].tolist()
        noise_radio = min(all_noise_radio)
        result_path = list()
        free_chanel = {}
        Path_final = ""

        for path in all_paths:
            if path[0] == input_node and path[len(path) - 1] == output_node:
                test = True
                final_occupancy = all_occupancy[all_paths_occupancy.index(path)]
                for i in range(len(final_occupancy)):
                    if final_occupancy[i]:
                        test = True
                        free_chanel[path] = i  # faire un dictionnaire qui envoie le path avec le channel libre
                        break
                    else:
                        test = False

                if all_noise_radio[all_paths.index(path)] > noise_radio and test is True:
                    noise_radio = all_noise_radio[all_paths.index(path)]
                    Path_final = path

        channel = free_chanel.get(Path_final)  # get the channel value for the best snr path

        result_path.append(Path_final)
        result_path.append(channel)

        return result_path

    def find_best_latency(self, input_node, output_node):
        Dataframe = self._weighted_paths
        Dataframe_occupancy = self._route_space
        all_paths = Dataframe["Paths"].tolist()
        all_latency = Dataframe["Latency (s)"].tolist()
        all_paths_occupancy = Dataframe_occupancy['Paths'].tolist()
        all_occupancy = Dataframe_occupancy['availability'].tolist()
        latency = max(all_latency)
        result_path = list()
        free_chanel = {}
        Path_final = ""

        for path in all_paths:
            if path[0] == input_node and path[len(path) - 1] == output_node:
                test = True
                final_occupancy = all_occupancy[all_paths_occupancy.index(path)]
                for i in range(len(final_occupancy)):
                    if final_occupancy[i]:
                        test = True
                        free_chanel[path] = i  # faire un dictionnaire qui envoie le path avec le channel libre
                        break
                    else:
                        test = False

                if all_latency[all_paths.index(path)] < latency and test is True:
                    latency = all_latency[all_paths.index(path)]
                    Path_final = path

        channel = free_chanel.get(Path_final)  # get the channel value for the best snr path

        result_path.append(Path_final)
        result_path.append(channel)

        return result_path

    def stream(self, connection, label="latency"):
        input = connection.input
        output = connection.output
        signal_power = connection.signal_power

        if label == "snr":
            path_snr = self.find_best_snr(input, output)
            final_path_snr = path_snr[0]
            final_path_snr = list(final_path_snr.split("->"))
            freq_channel = path_snr[1]

            if final_path_snr != ['']:
                lightpath = Lightpath(freq_channel, signal_power, final_path_snr)
                bit_rate = self.calculate_bit_rate(lightpath, self._nodes.get(
                    final_path_snr[0]).transceiver)

                if bit_rate != 0 or bit_rate is not None:
                    propagate_snr = self.propagate(lightpath)
                    connection.snr = 10 * math.log10(propagate_snr.signal_power / propagate_snr.noise_power)
                    connection.bit_rate = bit_rate

                else:
                    print("Connection rejected, path does not meet GSNR requirements")
                    connection.snr = 0

            else:
                connection.snr = 0

        elif label == "latency":
            path_latency = self.find_best_latency(input, output)
            final_path_latency = path_latency[0]
            final_path_latency = list(final_path_latency.split("->"))
            freq_channel = path_latency[1]

            if final_path_latency != ['']:
                signal_information = Lightpath(freq_channel, signal_power, final_path_latency)
                propagate_latency = self.propagate(signal_information)
                connection.latency = propagate_latency.latency
            else:
                connection.latency = 'None'

    def chanel_availability(self):
        var = list()
        path = list()
        switching_matrix = list()

        for i in self._nodes:
            for j in self._nodes:
                if not (var.__contains__((i, j)) or i == j):
                    var.append((i, j))
                    path.append(self.find_paths(i, j))

        result_data = list()

        for AllPaths in path:
            for actualPath in AllPaths:
                path_cpy = actualPath[:]  # keep the value of the patb
                path_cpy = '->'.join(path_cpy)  # we remove ->
                self.probe(Signal_information(0.001, actualPath))

                result_data.append(list([path_cpy]))  # we add each path in result_data
                final_path = path_cpy.split("->")  # we remove ->
                switching_matrix.append(self.update_route_space(final_path))

        data = {

            "Paths": [i[0] for i in result_data],
            "availability": [i for i in switching_matrix],

        }
        df = pd.DataFrame(data)
        # print(tabulate(df, showindex=True, headers=df.columns))

        return df

    def update_route_space(self, path):

        route_space_update = [1] * 10
        for i in range(len(path) - 2):
            route_space_update = [i1 * i2 for i1, i2 in zip(
                route_space_update, self._nodes.get(path[i + 1]).switching_matrix[path[i]][path[i + 2]])]

        for i in range(len(path) - 1):
            route_space_update = [i1 * i2 for i1, i2 in zip(
                route_space_update, self._lines.get(
                    path[i] + path[i + 1]).state)]  # On multiplie le précédent résultat par chaque line du path

        return route_space_update

    def calculate_bit_rate(self, lightpath, strategy):
        path_list = '->'.join(lightpath.path)  # change the list of string in A->B->C
        filtered_snr = self._weighted_paths.query(
            "Paths == @path_list")  # check if the path parameter is in the dataframe
        snr = filtered_snr.iloc[0]['Signal/noise (dB)']  # get the value of the gnsr according to the path
        ber_t = 10 ** -3
        rs = lightpath.Rs  # symbol rate
        bn = 12.5 * 10 ** 9  # noise bandwidth
        gsnr = 10 ** (snr / 10)

        if strategy == "fixed_rate":
            if gsnr >= 2 * erfcinv(2 * ber_t) ** 2 * rs / bn:
                return 100 * 10 ** 9  # return 100 Gbps
            else:
                return 0  # return 0 Gbps

        elif strategy == "flex_rate":
            if gsnr < 2 * erfcinv(2 * ber_t) ** 2 * rs / bn:
                return 0  # return 0 Gbps
            elif 2 * erfcinv(2 * ber_t) ** 2 * rs / bn <= gsnr < 14 / 3 * erfcinv(3 / 2 * ber_t) ** 2 * rs / bn:
                return 100 * 10 ** 9  # return 100 Gbps
            elif 14 / 3 * erfcinv(3 / 2 * ber_t) ** 2 * rs / bn <= gsnr < 10 * erfcinv(8 / 3 * ber_t) ** 2 * rs / bn:
                return 200 * 10 ** 9  # return 200 Gbps
            elif gsnr >= 10 * erfcinv(8 / 3 * ber_t) ** 2 * rs / bn:
                return 400 * 10 ** 9  # return 400 Gbps

        elif strategy == "shannon":
            return 2 * rs * math.log2(1 + (gsnr * rs / bn))  # return the maximum theoretical Shannon rate

    @staticmethod
    def histogram_accepted_connections(bit_rates):
        if len(bit_rates) > 0:
            total_capacity = sum(bit_rates)
            print("Total capacity allocated into the network {}".format(total_capacity * 10 ** 9))
            average_bit_rates = mean(bit_rates)
            plt.hist(bit_rates, bins=100)
            plt.xlabel('Bit Rate (Gbps)')
            plt.ylabel('Count')
            plt.title('Histogram of Accepted Connection Bit Rates | Average bit rates {}'.
                      format(round(average_bit_rates, 2)))  # return 2 number of the coma ( round)
            plt.show()
        else:
            print("All path does not reach the minimum GSNR requirement for the "
                  "specified transceiver strategy ")

    def traffic_matrix(self, type_method, numberConnection, M):

        # Create the traffic matrix
        traffic_matrix = {}
        for label in self.nodes:
            traffic_matrix[label] = {}
            for inner_label in self.nodes:
                bit_rate_request = 100e9 * M
                if label == inner_label:
                    bit_rate_request = 0
                traffic_matrix[label][inner_label] = bit_rate_request

        signal_power = 0.001
        connection = []
        node_result = []
        passed_con = 0  # count the number of passed connection
        rejected_con = 0  # count the number of rejected connection
        refused_requests = []

        print('Traffic matrix generation: ')
        print(traffic_matrix)
        traffic_matrix_tmp = copy.copy(traffic_matrix)

        while bool(traffic_matrix_tmp):
            for i in range(numberConnection):
                for keys in list(traffic_matrix):
                    for inner_keys in list(traffic_matrix[keys]):
                        if traffic_matrix[keys][inner_keys] <= 0 or (keys, inner_keys) in refused_requests:
                            del traffic_matrix_tmp[keys][inner_keys]
                        if bool(traffic_matrix_tmp[keys]) is False:  # remove case with no more output node
                            del traffic_matrix_tmp[keys]

                if bool(traffic_matrix_tmp):
                    input_node = random.choice(list(traffic_matrix_tmp.keys()))  # Random input node
                    output_node = random.choice(list(traffic_matrix_tmp[input_node].keys()))  # Random output node

                    connections = Connection(input_node, output_node, signal_power)
                    self.stream(connections, type_method)

                    if connections.bit_rate == 0:
                        print('Rejected connection ', input_node, '->', output_node,
                              '  --  Traffic request not supported')
                        refused_requests.append((input_node, output_node))
                        rejected_con += 1
                    else:
                        # update traffic matrix
                        traffic_matrix[input_node][output_node] -= connections.bit_rate
                        passed_con += 1

                        if traffic_matrix[input_node][output_node] <= 0:
                            traffic_matrix[input_node][output_node] = 0

                        connection.append(connections)
                else:
                    print('No more traffic after ', i, ' connections')
                    break

        print(traffic_matrix)
        print('Total rejected connection: ', rejected_con, ' over ', numberConnection)

        node_result.append(passed_con)
        node_result.append(connection)

        return node_result

    def all_result(self, type_method, connection_requests, M=None):

        M = M if M else 1  # 1 by default

        bit_rates = list()
        return_node = self.traffic_matrix(type_method, connection_requests, M)
        number_succeed_connection = return_node[0]  # number of validate nodes
        connections = return_node[1]  # all instances of connections

        if number_succeed_connection > 0:
            for i in range(number_succeed_connection):
                if connections[i].bit_rate is not None:
                    bit_rates.append(connections[i].bit_rate * 10 ** -9)

        self.histogram_accepted_connections(bit_rates)
