from Lightpath import Lightpath
import math
import scipy.constants as constant


class Line:

    def __init__(self, label, length, number_of_channel=10):
        self._label = label
        self._length = length
        self._successive = dict()
        self._number_of_channel = number_of_channel
        self._state = [True] * number_of_channel
        self._n_amplifiers = math.ceil(self._length / 80000) + 1
        self._gain_db = 16  # in dB
        self._gain = pow(10, self._gain_db / 10)
        self._noise_figure_db = 5.5  # in dB
        self._noise_figure = pow(10, self._noise_figure_db / 10)
        self._alpha_db = 0.2 * (self._length / 1000)
        self._alpha = self._alpha_db / (20 * math.log10(math.exp(1)))
        self._beta2 = 2.13e-26 * (self._length / 1000)
        self._gamma = 1.27e-3 * self.length
        self._optimal_launch_power = 0

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def successive(self):
        return self._successive

    @property
    def optimal_launch_power(self):
        return self._optimal_launch_power

    @label.setter
    def label(self, value):
        self._label = value

    @length.setter
    def length(self, length):
        self._length = length

    @successive.setter
    def successive(self, value):
        self._successive = value

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    @property
    def n_amplifiers(self):
        return self._n_amplifiers

    @property
    def gain(self):
        return self._gain

    @property
    def noise_figure(self):
        return self._noise_figure

    @n_amplifiers.setter
    def n_amplifiers(self, n_amplifiers):
        self._n_amplifiers = n_amplifiers

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    @noise_figure.setter
    def noise_figure(self, noise_figure):
        self._noise_figure = noise_figure

    @optimal_launch_power.setter
    def optimal_launch_power(self, power):
        self._optimal_launch_power = power

    def latency_generation(self):
        result = self._length / ((2 / 3) * 299792458)
        return result

    def noise_generation(self, signal_power):
        # return pow(10, -9) * signal_power * self._length
        ase = Line.ase_generation(self)
        nli = Line.nli_generation(self)
        noise = ase + nli
        return noise

    def propagate(self, lightpath: Lightpath):
        print(lightpath.channel)
        self._state[lightpath.channel] = 0  # we change the actual line with channel between 1 and 10 to False
        lightpath.update_noise_power(self.noise_generation(lightpath.signal_power))
        lightpath.UpdateLatency(self.latency_generation())
        return self._successive.get(lightpath.path[1]).propagate(lightpath)

    def probe(self, signal_information):
        signal_information.update_noise_power(self.noise_generation(signal_information.signal_power))
        signal_information.UpdateLatency(self.latency_generation())
        return self._successive.get(signal_information.path[0]).probe(signal_information)

    @property
    def number_of_channel(self):
        return self._number_of_channel

    def ase_generation(self):
        frequency = 193.414e12
        Bn = 12.5e9
        return self._n_amplifiers * (constant.Planck * frequency * Bn * self._noise_figure * (self._gain - 1))

    def nli_generation(self):
        Rs = 32e9
        df = 50e9
        frequency = 193.414e12
        Leff = 1 / (2 * self._alpha)
        n_span = self._n_amplifiers - 1
        Bn = 12.5e9
        n_nli = (16 / (27 * math.pi))
        math.log10((pow(math.pi, 2) * self._beta2 * pow(Rs, 2) * pow(10, (2 * Rs) / df)) / (2 * self._alpha)) * (
                    (self._alpha * pow(self._gamma, 2) * pow(Leff, 2)) / (self._beta2 * pow(Rs, 3)))
        p_ch = ((constant.Planck * frequency * Bn * self._noise_figure * self._alpha * self.length) / (
                    2 * Bn * n_nli)) ** (1 / 3)
        return pow(p_ch, 3) * n_nli * n_span * Bn

    def optimized_launch_power(self):
        ase = self.ase_generation()
        nli = self.nli_generation()
        optimal_launch_power = (ase / (2 * nli)) ** (1 / 3)  # Pch
        return optimal_launch_power
