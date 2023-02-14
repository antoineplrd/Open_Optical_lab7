class Node:

    def __init__(self, label, position, connected_nodes, switching_matrix, transceiver="fixed_rate"):
        self._label = label
        self._position = tuple(position)
        self._connected_nodes = connected_nodes
        self._successive = dict()
        self._switching_matrix = switching_matrix
        self._transceiver = transceiver

    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @property
    def successive(self):
        return self._successive

    @property
    def switching_matrix(self):
        return self._switching_matrix

    @property
    def transceiver(self):
        return self._transceiver

    @label.setter
    def label(self, value):
        self._label = value

    @position.setter
    def position(self, value):
        self._position = value

    @connected_nodes.setter
    def connected_nodes(self, value):
        self._connected_nodes = value

    @successive.setter
    def successive(self, value):
        self._successive = value

    @switching_matrix.setter
    def switching_matrix(self, switching_matrix):
        self._switching_matrix = switching_matrix

    @transceiver.setter
    def transceiver(self, transceiver):
        self._transceiver = transceiver

    def propagate(self, signal_information, is_first=False):
        path = signal_information.path
        if not is_first:
            if len(path) > 2:
                if signal_information.channel > 1:
                    self._switching_matrix[path[0]][path[2]][
                        signal_information.channel - 1] = 0  # we set the previous index channel to 0
                if signal_information.channel < 9:
                    self._switching_matrix[path[0]][path[2]][
                        signal_information.channel + 1] = 0  # we set the next index channel to 0
                self._switching_matrix[path[0]][path[2]][
                    signal_information.channel] = 0  # we set the index channel to 0
            signal_information.UpdatePath_CrossedNode()  # call crossed node to remove current node
        if len(path) > 1:
            line = self._successive.get(path[0] + path[1])  # get the successive line
            signal_information.signal_power = line.optimized_launch_power()
            line = line.propagate(signal_information)

            return line
        else:
            signal_information.UpdatePath_CrossedNode()  # call crossed node to remove current node
            return signal_information  # return the signal information

    def probe(self, signal_information):
        path = signal_information.path[:]
        signal_information.UpdatePath_CrossedNode()
        if len(path) > 1:

            return self._successive.get(path[0] + path[1]).probe(signal_information)

        else:
            return signal_information
