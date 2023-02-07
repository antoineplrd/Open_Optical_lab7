import math
import scipy.constants as constant
import numpy as np

from Lightpath import Lightpath


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
        # self._alpha = self.alpha_db / (20 * math.log10(math.exp(1)))
        self._alpha = 0.2 / (20 * np.log10(np.exp(1)) * 1000)  # alpha = 0.2 dB/Km
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
    def gain_db(self):
        return self._gain_db

    @property
    def noise_figure_db(self):
        return self._noise_figure_db

    @n_amplifiers.setter
    def n_amplifiers(self, n_amplifiers):
        self._n_amplifiers = n_amplifiers

    @gain_db.setter
    def gain_db(self, gain_db):
        self._gain_db = gain_db

    @property
    def number_of_channel(self):
        return self._number_of_channel

    @property
    def alpha_db(self):
        return self._alpha_db

    @noise_figure_db.setter
    def noise_figure_db(self, noise_figure_db):
        self._noise_figure_db = noise_figure_db

    def latency_generation(self):
        result = self._length / ((2 / 3) * 299792458)
        return result

    @property
    def optimal_launch_power(self):
        return self._optimal_launch_power

    @optimal_launch_power.setter
    def optimal_launch_power(self, power):
        self._optimal_launch_power = power

    def noise_generation(self, signal_power):
        # return pow(10, -9) * signal_power * self._length
        ase = self.ase_generation()
        nli = self.nli_generation()
        noise = ase + nli
        return noise

    def propagate(self, lightpath: Lightpath):
        self._state[lightpath.channel] = False  # we change the actual line with channel between 1 and 10 to False
        lightpath.update_noise_power(self.noise_generation(lightpath.signal_power))
        lightpath.UpdateLatency(self.latency_generation())
        return self._successive.get(lightpath.path[1]).propagate(lightpath)

    def probe(self, signal_information):
        signal_information.update_noise_power(self.noise_generation(signal_information.signal_power))
        signal_information.UpdateLatency(self.latency_generation())
        return self._successive.get(signal_information.path[0]).probe(signal_information)

    def ase_generation(self):
        frequency = 193.414e12
        Bn = 12.5e9
        return self._n_amplifiers * (constant.Planck * frequency * Bn * self._noise_figure * (self._gain - 1))

    def nli_generation(self):
        Rs = 32e9
        df = 50e9
        frequency = 193.414e12
        n_span = self._n_amplifiers - 1
        Bn = 12.5e9
        x1 = 0.5 * (np.pi ** 2) * self._beta2 * (Rs ** 2) * (1 / self._alpha) * 10 ** (2 * (Rs / df))
        x2 = (self._gamma ** 2) / (4 * self._alpha * self._beta2 * (Rs ** 3))
        eta_nli = (16 / (27 * np.pi)) * np.log10(x1) * x2
        p_ch = ((constant.Planck*frequency*Bn*self._noise_figure*self._alpha*self.length)/(2*Bn*eta_nli))**(1/3)
        return pow(p_ch, 3) * eta_nli * n_span * Bn

    def optimized_launch_power(self):
        n_span = self._n_amplifiers - 1
        Bn = 12.5e9
        ase = self.ase_generation()
        nli = self.nli_generation()
        optimal_launch_power = (ase / ((2 * nli) * n_span * Bn)) ** (1 / 3)
        return optimal_launch_power
