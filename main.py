# Lab 7 - Open Optical Network ANTOINE POUILLARD
from Network import Network


def main():
    print("######### FIXED RATE ###########")

    network1 = Network("nodes_full_fixed_rate.json")
    fixed = network1.all_result('snr', 100, 5)  # 100 connection with M of 5

    print("######### FLEX RATE ###########")
    network2 = Network("nodes_full_flex_rate.json")
    flex = network2.all_result('snr', 100, 5)

    print("######### SHANNON ###########")
    network3 = Network("nodes_full_shannon.json")
    shannon = network3.all_result('snr', 100, 5)

    ###### GRAPH REPRESENTATION #########
    network1.draw()
    bit_rates = [fixed[0], flex[0], shannon[0]]
    Network.histogram_bit_rates_all_tranceiver(bit_rates)  # bit rate value for all tranceiver
    Network.graph_capacity_allocated_all_tranceiver(bit_rates)  # capacity in the network for all tranceiver
    # Network.histogram_accepted_connections(bit_rates[0])  # all the bit rate values for a specific tranceiver

    """fixed_increasing_M = []
    for i in range(10):
        print("######### FIXED RATE ###########")
        network1 = Network("nodes_full_fixed_rate.json")
        fixed_temp = network1.all_result('snr', 100, i)[0]
        fixed_increasing_M.append(sum(fixed_temp))  # 100 connection with M of 5
    Network.graph_capacity_allocated(fixed_increasing_M)  # capacity in the network with an increasing M for a particular tranceiver"""

    # traffic_matrix = fixed[1]  # get the trafic matrix for fixed tranceiver
    # Network.graph_trafic_matrix(traffic_matrix)  # print trafic matrix for a particular tranceiver

    network1.graph_SNR()  # snr graph for fixed rate tranceiver


if "__main__" == __name__:
    main()
