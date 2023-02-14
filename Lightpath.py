from Signal_information import Signal_information


class Lightpath(Signal_information):
    def __init__(self, channel, signal_power, path):
        super().__init__(signal_power, path)  # super because lightpath is a child of signal information
        self._channel = channel
        self._Rs = 32e9
        self._df = 50e9

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        self._channel = value

    @property
    def Rs(self):
        return self._Rs

    @Rs.setter
    def Rs(self, Rs):
        self._Rs = Rs

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df):
        self._df = df
