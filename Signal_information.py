class Signal_information:

    def __init__(self, signal_power, path):
        self._signal_power = signal_power
        self._noise_power = 0
        self._latency = 0
        self._path = path

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def noise_power(self):
        return self._noise_power

    @property
    def latency(self):
        return self._latency

    @property
    def path(self):
        return self._path

    @signal_power.setter
    def signal_power(self, x):
        self._signal_power = x

    @noise_power.setter
    def noise_power(self, x):
        self._noise_power = x

    @latency.setter
    def latency(self, x):
        self._latency = x

    @path.setter
    def path(self, x):
        self._path = x

    def UpdatePath_CrossedNode(self):
        if self._path is not None:

            del self._path[0]

    def UpdateLatency(self, latency):
        self._latency += latency

    def update_noise_power(self, noise_power):
        self._noise_power += noise_power
