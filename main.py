# Lab 4 - Open Optical Network ANTOINE POUILLARD
import networkx as nx
from matplotlib import pyplot as plt
import string
import random
from Network import Network
from Connection import Connection
from tabulate import tabulate


def main():
    network = Network("nodes_full_fixed_rate.json")
    nodeValue = 'ABCDEF'
    signal_power = 0.001
    bit_rates = list()
    for i in range(0, 100):
        inputNode = random.choice(nodeValue)
        outputNode = random.choice(nodeValue)
        while inputNode == outputNode:  # if we have the same node
            inputNode = random.choice(nodeValue)
            outputNode = random.choice(nodeValue)
            if inputNode != outputNode:
                break

        connections = Connection("A", "B", signal_power)

        # network.stream(connections, 'latency')
        network.stream(connections, 'snr')
    # network.draw() # modifier pour mettre le chemin ?

    if connections.bit_rate is not None:
        bit_rates.append(connections.bit_rate * 10 ** -9)

        # network.draw()
    network.histogram_accepted_connections(bit_rates)


if "__main__" == __name__:
    main()
