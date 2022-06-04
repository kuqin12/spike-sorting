import wave
from matplotlib import pyplot as plt
import numpy as np

from numpy import savetxt, loadtxt

from scipy.interpolate import interp1d
from scipy import interpolate

import random

# 1. Generate a basis of synthetic spikes from 3 different neurons

n1 = np.array(
    [0,0.02,0.04,0.1,0.2,
    0.55,0.95,1.0,0.95,0.55,
    0.15,-0.15,-0.55,-0.95,-1.0,
    -0.95,-0.55,-0.1,0.05,0.1,
    0.18,0.2,0.18,0.12,0.1,
    0.08,0.07,0.06,0.05,0.02,
    0,0,0,0,0,
    0,0,0,0,0])

n2 = np.array(
    [0,0.02,0.04,0.1,0.2,
    0.25,0.3,0.33,0.3,0.25,
    0.2,0.1,0.04,0.02,0,
    -0.02,-0.04,-0.1,-0.2,-0.23,
    -0.2,-0.18,-0.14,-0.1,-0.02,
    -0.01,-0.005,0,0,0,
    0,0,0,0,0,
    0,0,0,0,0]
)

n3 = np.array(
    [0,-0.02,-0.04,-0.1,-0.2,
    -0.45,-0.5,-0.45,-0.35,-0.25,
    0,0.2,0.45,0.75,0.95,
    1.0,0.95,0.75,0.55,0.3,
    0.18,0,-0.18,-0.22,-0.18,
    -0.1,-0.05,0,0,0,
    0,0,0,0,0,
    0,0,0,0,0]
)

x = np.array(range(40))

n1_smooth = interpolate.splev(x, interpolate.splrep(x, n1), der=0)
n2_smooth = interpolate.splev(x, interpolate.splrep(x, n2), der=0)
n3_smooth = interpolate.splev(x, interpolate.splrep(x, n3), der=0)

PLOT_WAVEFORMS = False
if PLOT_WAVEFORMS:
    plt.plot(x, n1, '*', x, n1_smooth, '-')
    plt.plot(x, n2, 'o', x, n2_smooth, '-')
    plt.plot(x, n3, 'o', x, n3_smooth, '-')
    plt.legend(['data', 'spline'], loc='best')
    plt.show(block=False)
    plt.pause(10)
    plt.close()

basis = [ n1_smooth, n2_smooth, n3_smooth ]

# 2 Generate n_copies of each spike with added noise

n_copies = 10
noise = 0.1

def get_sample_spikes ():
    noisy_spikes = []

    for spike in basis:
        for i in range(n_copies):
            # Generate noise and keep between -1 and 1
            noisy_spike = [ sample * (1 + random.uniform(-noise,noise)) for sample in spike ]
            noisy_spike = [ sample if sample <= 1.0 else 1.0 for sample in noisy_spike ]
            noisy_spike = [ sample if sample >= -1.0 else -1.0 for sample in noisy_spike ]
            noisy_spikes.append(noisy_spike)

    SHOW_NOISY_SPIKES = False
    if SHOW_NOISY_SPIKES:
        for spike in noisy_spikes:
            plt.plot(x, spike)

        plt.show(block=False)
        plt.pause(5)
        plt.close()

    return noisy_spikes
