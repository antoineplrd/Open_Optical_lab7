# Lab 7 - Open Optical Network ANTOINE POUILLARD
from matplotlib import pyplot as plt
from Network import Network
from Connection import Connection


def main():
    """print("######### FIXED RATE ###########")
    network1 = Network("nodes_full_fixed_rate.json")
    network1.all_result('snr', 100, 5)  # 100 connection with M of 5"""

    print("######### FLEX RATE ###########")
    network2 = Network("nodes_full_flex_rate.json")
    network2.all_result('snr', 100, 4)

    """print("######### SHANNON ###########")
    network3 = Network("nodes_full_shannon.json")
    network3.all_result('snr', 100, 5)"""


if "__main__" == __name__:
    main()
