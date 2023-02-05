class Connection:

    def __init__(self, input, output, signal_power):
        self._input = input
        self._output = output
        self._signal_power = signal_power
        self._latency = 0
        self._snr = 0
        self._bit_rate = None

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, input):
        self._input = input

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, output):
        self._output = output

    @property
    def signal_power(self):
        return self._signal_power

    @signal_power.setter
    def signal_power(self, signal_power):
        self._signal_power = signal_power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, snr):
        self._snr = snr

    @property
    def bit_rate(self):
        return self._bit_rate

    @bit_rate.setter
    def bit_rate(self, bit_rate):
        self._bit_rate = bit_rate
