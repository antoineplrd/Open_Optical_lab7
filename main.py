# Lab 7 - Open Optical Network ANTOINE POUILLARD
from matplotlib import pyplot as plt
from Network import Network
from Connection import Connection


def main():
    network = Network("nodes_full_flex_rate.json")
    nodeValue = 'ABCDEF'
    signal_power = 0.001
    bit_rates = list()

    return_node = network.traffic_matrix(100)
    number_connection = return_node[2]  # number of validate nodes

    if number_connection > 0:
        for i in range(number_connection):
            inputNode = return_node[0][i]
            outputNode = return_node[1][i]

            connections = Connection(inputNode, outputNode, signal_power)

            # network.stream(connections, 'latency')
            network.stream(connections, 'snr')
            if connections.bit_rate is not None:
                bit_rates.append(connections.bit_rate * 10 ** -9)

    # network.draw()
    network.histogram_accepted_connections(bit_rates)
    # print(tabulate(df, showindex=True, headers=df.columns))


if "__main__" == __name__:
    main()
