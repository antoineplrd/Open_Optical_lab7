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

            for k in data[i]["position"]:  # get position from json file
                connected_lines.append(k)
            for j in data[i]["connected_nodes"]:  # get connected nodes from json file
                # implementation for line between 2 nodes: AB, BA
                connected_nodes.append(j)

            if "switching_matrix" in data[i]:  # case when whe don't have a transceiver in the json file
                switching_matrix = data[i]["switching_matrix"]
            else:
                switching_matrix = None

            if "transceiver" in data[i]:  # case when whe don't have a transceiver in the json file
                transceiver = data[i]["transceiver"]
            else:
                transceiver = None

            node = Node(label, (connected_lines[0], connected_lines[1]), connected_nodes,
                        switching_matrix, transceiver)  # create a instance of node
            self._nodes.update({label: node})
            # print(self._nodes.get(i).switching_matrix) print of the switching matrix for each node

        open_Json.close()  # close json

        for i in self._nodes:
            # length between 2 nodes
            labelX = self._nodes.get(i).position[0]
            labelY = self._nodes.get(i).position[1]
            for j in self._nodes.get(i).connected_nodes:
                label_lines = i + j
                nextNodeX = self._nodes.get(j).position[0]
                nextNodeY = self._nodes.get(j).position[1]
                Distance_lines = sqrt((nextNodeX - labelX) ** 2 + (nextNodeY - labelY) ** 2)  # get the distance

                line = Line(label_lines, Distance_lines)  # create a instance of line
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
            self.lines.get(i).successive = successive_nodes  # we update the successive line for each two nodes
        for i in self._nodes:
            successive_lines = dict()
            for j in self._nodes.get(i).connected_nodes:
                successive_lines.update({i + j: self._lines.get(i + j)})
            self._nodes.get(i).successive = successive_lines

    def find_paths(self, start_node, end_node, path=None):

        if path is None:  # case path is not empty
            path = []
        graph_dict = self._nodes.get(start_node).connected_nodes  # get all the connected node to the start node
        path = path + [start_node]  # we add each time the current start node
        if start_node == end_node:  # case when we have the same input and output node
            return [path]
        paths = []
        for actual_node in graph_dict:
            if actual_node not in path:
                extended_paths = self.find_paths(actual_node, end_node, path)  # call recurcive function to get path
                for p in extended_paths:
                    paths.append(p)
        return paths

    def propagate(self, signal_information):
        propagate = self._nodes.get(signal_information.path[0]).propagate(signal_information, True)  # propagate node
        self._route_space = self.chanel_availability()  # We update for each new path the avability
        return propagate

    def probe(self, signal_information):
        return self._nodes.get(signal_information.path[0]).probe(signal_information)

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
                path_cpy = '->'.join(path_cpy)  # we remove the ->
                signal_information = self.probe(Signal_information(0.001, j))
                ratio = 10 * math.log10(
                    signal_information.signal_power / signal_information.noise_power)  # ratio formula
                test.append(list([path_cpy, signal_information.latency, signal_information.noise_power, ratio]))

        data = {

            "Paths": [i[0] for i in test],
            "Latency (s)": [i[1] for i in test],
            "Noise power (W)": [i[2] for i in test],
            "Signal/noise (dB)": [i[3] for i in test]
        }
        pd.set_option('display.max_columns', None)  # two next line to print all the line of the dataframe
        pd.set_option('display.max_rows', None)
        df = pd.DataFrame(data)

        return df

    def find_best_snr(self, input_node, output_node):
        Dataframe = self._weighted_paths  # get the weighted paths dataframe
        Dataframe_occupancy = self._route_space  # get the route space dataframe
        all_paths = Dataframe['Paths'].tolist()  # get the paths
        all_noise_radio = Dataframe['Signal/noise (dB)'].tolist()  # get the SNR
        all_paths_occupancy = Dataframe_occupancy['Paths'].tolist()  # get the paths
        all_occupancy = Dataframe_occupancy['availability'].tolist()  # get the availability for each path
        noise_radio = min(all_noise_radio)  # start with the miminum SNR
        result_path = list()
        free_chanel = {}
        Path_final = ""

        for path in all_paths:
            if path[0] == input_node and path[
                len(path) - 1] == output_node:  # check if the input and output node are correct
                test = True
                final_occupancy = all_occupancy[all_paths_occupancy.index(path)]  # get the availabity for specific path
                for i in range(len(final_occupancy)):
                    if final_occupancy[i]:  # if it is true
                        test = True
                        free_chanel[path] = i  # send in free chanel the index of the free chanel
                        break
                    else:
                        test = False

                if all_noise_radio[
                    all_paths.index(path)] > noise_radio and test is True:  # find the highest noise radio
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
                        free_chanel[path] = i
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
        input = connection.input  # input node
        output = connection.output  # output node
        signal_power = connection.signal_power

        if label == "snr":  # case when we use SNR
            path_snr = self.find_best_snr(input, output)  # find the best path with highest SNR (return a list)
            final_path_snr = path_snr[0]  # get the path
            final_path_snr = list(final_path_snr.split("->"))  # we remove ->
            freq_channel = path_snr[1]  # get the free channel index

            if final_path_snr != ['']:  # check if the path in not empty
                lightpath = Lightpath(freq_channel, signal_power, final_path_snr)  # create a lightpath
                bit_rate = self.calculate_bit_rate(lightpath, self._nodes.get(
                    final_path_snr[0]).transceiver)  # calculate the bit rate value for the path

                if bit_rate != 0 or bit_rate is not None:  # case when bit rate is not null or equal to 0
                    propagate_snr = self.propagate(lightpath)  # we propagate the lighpath
                    connection.snr = 10 * math.log10(
                        propagate_snr.signal_power / propagate_snr.noise_power)  # set the snr
                    connection.bit_rate = bit_rate  # set the bit rate

                else:
                    print("Connection rejected, path does not meet GSNR requirements")
                    connection.snr = 0

            else:
                connection.snr = 0

        elif label == "latency":  # case when we use the latency
            path_latency = self.find_best_latency(input, output)  # find the best path with smallest latency
            final_path_latency = path_latency[0]  # get the best path ( path latency return a list)
            final_path_latency = list(final_path_latency.split("->"))  # we remove ->
            freq_channel = path_latency[1]  # get the

            if final_path_latency != ['']:  # case when the path is not empty
                lightpath = Lightpath(freq_channel, signal_power, final_path_latency)  # create instance of lightpath
                propagate_latency = self.propagate(lightpath)  # propagate the lightpath
                connection.latency = propagate_latency.latency  # set the latency
            else:
                connection.latency = 'None'

    def chanel_availability(self):
        var = list()
        path = list()
        switching_matrix = list()

        for i in self._nodes:
            for j in self._nodes:
                if not (var.__contains__((i, j)) or i == j):  # test if node have been already test

                    var.append((i, j))
                    path.append(self.find_paths(i, j))  # send the path

        result_data = list()

        for AllPaths in path:  # go throught all the pat
            for actualPath in AllPaths:
                path_cpy = actualPath[:]  # keep the value of the patb
                path_cpy = '->'.join(path_cpy)  # we remove ->
                self.probe(Signal_information(0.001, actualPath))

                result_data.append(list([path_cpy]))  # we add each path in result_data
                final_path = path_cpy.split("->")  # we remove ->
                switching_matrix.append(
                    self.update_route_space(final_path))  # we set the switching matrix calling update route space

        data = {

            "Paths": [i[0] for i in result_data],
            "availability": [i for i in switching_matrix],

        }
        df = pd.DataFrame(data)
        # print(tabulate(df, showindex=True, headers=df.columns))

        return df

    def update_route_space(self, path):

        route_space_update = [1] * 10  # starting setting each line to 1 (free)
        for i in range(len(path) - 2):  # first case where path greater than 2 nodes
            route_space_update = [i1 * i2 for i1, i2 in zip(
                route_space_update, self._nodes.get(path[i + 1]).switching_matrix[path[i]][path[i + 2]])]

        for i in range(len(path) - 1):  # for line who have more that 1 node
            route_space_update = [i1 * i2 for i1, i2 in zip(
                route_space_update, self._lines.get(
                    path[i] + path[i + 1]).state)]  # we multiplies the previous result by each line of the path

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

    def traffic_matrix(self, type_method, numberConnection, M):

        # Create the traffic matrix
        traffic_matrix = {}
        for label in self.nodes:
            traffic_matrix[label] = {}
            for inner_label in self.nodes:
                bit_rate_request = 100e9 * M  # create the bit request for each nodes who will depend on the M
                if label == inner_label:
                    bit_rate_request = 0  # put to zero for the same input and output node
                traffic_matrix[label][inner_label] = bit_rate_request  # set the previous value

        signal_power = 0.001
        connection = []
        node_result = []
        snr = []
        passed_con = 0  # count the number of passed connection
        rejected_con = 0  # count the number of rejected connection
        refused_requests = []

        print('Traffic matrix generation: ')
        traffic_matrix_tmp = copy.deepcopy(traffic_matrix)  # make a copy of the traffic matrix

        while bool(traffic_matrix_tmp):  # keep going when traffic matrix is True
            for i in range(numberConnection):
                for keys in list(traffic_matrix):  # line
                    for inner_keys in list(traffic_matrix[keys]):  # colum
                        if traffic_matrix[keys][inner_keys] <= 0 or (keys, inner_keys) in refused_requests:
                            if inner_keys in traffic_matrix_tmp.get(keys,
                                                                    {}):  # check if the key exist in the tmp matrix
                                del traffic_matrix_tmp[keys][inner_keys]  # we remove it from the traffic matrix copy
                        if keys in traffic_matrix_tmp:
                            if bool(traffic_matrix_tmp[keys]) is False:  # remove case with no more output node
                                del traffic_matrix_tmp[keys]  # delete in the temp matrix

                if bool(traffic_matrix_tmp):  # keep going when traffic matrix is True
                    input_node = random.choice(list(traffic_matrix_tmp.keys()))  # Random input node
                    output_node = random.choice(list(traffic_matrix_tmp[input_node].keys()))  # Random output node

                    connections = Connection(input_node, output_node, signal_power)  # create a connection
                    self.stream(connections, type_method)  # call the stream function
                    snr.append(connections.snr)  # get the SNR value for the snr graph

                    if connections.bit_rate == 0:  # when the bit rate is 0
                        print('Rejected connection ', input_node, '->', output_node,
                              '  --  Traffic request not supported')  # we rejected connection
                        refused_requests.append((input_node, output_node))  # we add it to our list
                        rejected_con += 1  # we increment the number of rejected connection
                    else:
                        # update traffic matrix
                        traffic_matrix[input_node][
                            output_node] -= connections.bit_rate  # we reduce the traffic matrix bit rate
                        passed_con += 1

                        if traffic_matrix[input_node][
                            output_node] <= 0:  # if value < 0 , set it to 0 ( none negative value)
                            traffic_matrix[input_node][output_node] = 0

                        connection.append(connections)  # we send all the instance of connection for our next function
                else:
                    print('No more traffic after ', i,
                          ' connections')  # print the number of connection after when the traffic matrix is saturated
                    break

        print('Total rejected connection: ', rejected_con, ' over ', numberConnection)

        node_result.append(passed_con)
        node_result.append(connection)
        node_result.append(traffic_matrix)
        node_result.append(snr)

        return node_result

    def all_result(self, type_method, connection_requests, M=None):

        M = M if M else 1  # 1 by default

        bit_rates = list()
        result = []
        return_node = self.traffic_matrix(type_method, connection_requests, M)  # get the traffic matrix result (list)
        number_succeed_connection = return_node[0]  # number of validate nodes
        traffic_matrix = return_node[2]
        snr = return_node[3]  # get snr from the traffic_matrix function
        connections = return_node[1]  # all instances of connections

        if number_succeed_connection > 0:  # do this only if the number of connection is greater than 1
            for i in range(number_succeed_connection):
                if connections[i].bit_rate is not None:  # case where bit rate is not null
                    bit_rates.append(connections[i].bit_rate * 10 ** -9)  # send all the connection bit rate in a list
        result.append(bit_rates)  # return the list of bit rate
        result.append(traffic_matrix)  # return the traffic matrix after all the propagation
        result.append(snr)

        return result

    @staticmethod
    def histogram_accepted_connections(bit_rates):
        if len(bit_rates) > 0:
            plt.hist(bit_rates, bins=100)
            plt.xlabel('Bit Rate (Gbps)')
            plt.ylabel('Count')
            plt.title('Histogram of Accepted Connection Bit Rates')  # return 2 number of the coma ( round)
            plt.show()
        else:
            print("All path does not reach the minimum GSNR requirement for the "
                  "specified transceiver strategy ")

    @staticmethod
    def histogram_bit_rates_all_tranceiver(bit_rates):
        fixed = bit_rates[0]
        flex = bit_rates[1]
        shannon = bit_rates[2]

        plt.bar('fixed', fixed, color='green', width=0.6)
        plt.bar('Flex', flex, color='blue', width=0.6)
        plt.bar('shannon', shannon, color='red', width=0.6)

        plt.ylabel('Bit Rate (Gbps)')
        plt.title('Bit rate for different tranceiver with 100 runs  and M = 5 ')
        plt.show()

    @staticmethod
    def graph_trafic_matrix(traffic_matrix):

        rows = list(traffic_matrix.keys())
        columns = list(traffic_matrix[rows[0]].keys())

        fig, ax = plt.subplots()
        table_data = [[''] + columns]

        for row in rows:
            table_data.append([row] + [traffic_matrix[row][col] for col in columns])

        table = ax.table(cellText=table_data, cellLoc='center',
                         colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15], loc='center')
        table.auto_set_font_size(10)
        table.set_fontsize(14)

        ax.axis('off')
        plt.show()

    def graph_SNR(self):

        Dataframe = self._weighted_paths
        snr = Dataframe['Signal/noise (dB)'].tolist()

        plt.hist(snr, width=0.2, color='green', bins=100)
        plt.ylabel('Count', fontsize=12, color='gray')
        plt.xlabel('Signal/noise in Db', fontsize=12, color='gray')

        plt.title('SNR distribution', fontsize=16, color='red')

        plt.grid(linestyle='--', color='gray', alpha=0.7)
        plt.show()

    @staticmethod
    def graph_capacity_allocated_all_tranceiver(bit_rates):

        total_capacity_fixed = sum(bit_rates[0])
        total_capacity_flex = sum(bit_rates[1])
        total_capacity_shannon = sum(bit_rates[2])

        plt.bar('fixed', total_capacity_fixed, color='green', width=0.6)
        plt.bar('Flex', total_capacity_flex, color='blue', width=0.6)
        plt.bar('shannon', total_capacity_shannon, color='red', width=0.6)

        plt.ylabel('total_capacity (in 10^9)')
        plt.title('Total capacity allocated into the network with 100 runs  and M = 5 for each tranceiver ',
                  color='red')
        plt.grid(linestyle='--', color='gray', alpha=0.7)

        plt.show()

    @staticmethod
    def graph_capacity_allocated(bit_rates):

        # increasing value of m
        counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # tracer le graphique
        plt.plot(counts, bit_rates, '-o')

        plt.ylabel('total_capacity (in 10^9)')
        plt.xlabel('Value of M increasing')
        plt.title('Total capacity allocated into the network with 100 runs  and M increasing ', color='red')
        plt.grid(linestyle='--', color='gray', alpha=0.7)
        plt.show()

    def draw(self):
        G = nx.Graph()
        for i in self._nodes:
            G.add_nodes_from(self._nodes.get(i).label, pos=(self._nodes.get(i).position[0],
                                                            self._nodes.get(i).position[1]))
        for j in self._lines:
            G.add_edges_from([(j[0], j[1])])
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True)
        plt.ylabel('Position Y', fontsize=12, color='gray')
        plt.xlabel('Position X', fontsize=12, color='gray')

        plt.grid(linestyle='--', color='gray', alpha=0.7)
        plt.show()
