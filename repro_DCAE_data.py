from matplotlib import pyplot as plt

from scipy.interpolate import interp1d
from scipy import interpolate

import numpy as np

import random

#Define envelopes for spike waveform templates
M_upper = [(237,87),(252,94),(264,130),(268,176),(275,209),(279,176),(288,106),(295,60),(305,33),(317,25),(334,34),(350,54),(365,66),(380,76),(395,80),(405,82)]
M_lower = [(237,122),(252,141),(264,199),(268,232),(275,250),(279,231),(288,162),(295,122),(305,71),(317,55),(334,73),(350,87),(365,102), (380,109),(395,114),(405,118)]
B_upper = [(238,80),(246,64),(252,55),(260,70),(262,86),(276,169),(289,94),(299,76),(314,71),(333,75),(350,76),(381,78),(406,82)]
B_lower = [(238,123),(246,125),(252,156),(260,178),(262,180),(276,190),(289,156),(299,120),(314,108),(333,110),(350,115),(381,117),(406,116)]
P_upper = [(243,81),(252,70),(258,58),(263,40),(271,27),(280,42),(290,92),(300,128),(317,190),(331,186),(355,165),(366,153),(378,142),(397,131),(407,128)]
P_lower = [(243,124),(252,147),(258,178),(263,124),(271,71),(280,113),(290,196),(300,216),(317,232),(331,228),(355,205),(366,193),(378,184),(397,175),(407,170)]
O_upper = [(234,85),(256,90),(275,102),(283,124),(301,190),(318,143),(330,94),(339,82),(348,79),(365,77),(390,78),(407,83)]
O_lower = [(234,108),(256,106),(275,133),(283,172),(301,222),(318,181),(330,122),(339,102),(348,99),(365,102), (390,114),(407,104)]
G_upper = [(237,82),(248,74),(262,87),(278,91),(297,135),(305,132),(317,145),(330,170),(351,150),(370,134),(385,121),(406,118)]
G_lower = [(237,125),(248,134),(262,180),(278,185),(297,204),(305,215),(317,212),(330,211),(351,191),(370,177),(385,168),(406,156)]
Y_upper = [(238,81),(248,77),(258,87),(272,160),(274,172),(277,190),(292,161),(310,118),(322,92),(335,74),(358,60),(377,66),(393,69),(407,74)]
Y_lower = [(238,115),(248,103),(258,144),(272,215),(274,235),(277,241),(292,200),(310,157),(322,124),(335,114),(358,88),(377,87),(393,91),(407,97)]

class SpikeTemplate:

    SAMPLES_PER_WAVEFORM = 40
    SIGMA = 4

    def __init__(self, upper, lower):

        x  = [ u[0] for u in upper ]
        y1 = [ u[1] for u in upper ]
        y2 = [ l[1] for l in lower ]

        self.x  = np.array(x)
        self.y1 = -1*np.array(y1) 
        self.y2 = -1*np.array(y2)

    def generate(self, n, d='uniform'):

        self.spikes = []
        for i in range(n):

            new_spike = []

            self.super_resolution(times=8)

            for j in range(len(self.xx)):

                if d == 'uniform':
                    new_spike.append(random.uniform(self.yy2[j], self.yy1[j]))
                elif d == 'gaussian':
                    new_spike.append(np.random.normal(np.mean([self.yy2[j], self.yy1[j]]), self.SIGMA))

            new_spike = self.downsample(new_spike, self.SAMPLES_PER_WAVEFORM)
            
            self.spikes.append((np.array(new_spike)+280)/280)

    def super_resolution(self, times=1):

        #linearly interpolates two consecutive points
        xx = self.x
        yy1 = self.y1
        yy2 = self.y2 

        for i in range(times):

            xx = xx.repeat(2)
            xx = (xx[1:]+xx[:-1]) / 2

            yy1 = yy1.repeat(2)
            yy1 = (yy1[1:]+yy1[:-1]) / 2

            yy2 = yy2.repeat(2)
            yy2 = (yy2[1:]+yy2[:-1]) / 2

        self.xx = xx 
        self.yy1 = yy1 
        self.yy2 = yy2

    def downsample(self, spike, n_samples=40):

        out = []

        for i in range(0,n_samples):

            #Get the index relative to the template
            idx = (self.xx[-1] - self.xx[0]) / float(n_samples) * i + self.xx[0]
            idx2 = np.argmax(self.xx > idx)
            out.append(spike[idx2])

        return out

    def normalize(self, spike, min, max):
        pass
            
class SpikeGenerator():

    spikes = []

    def __init__(self):

        self.basis = [    
            SpikeTemplate(B_upper, B_lower), 
            SpikeTemplate(P_upper, P_lower),
            SpikeTemplate(G_upper, G_lower),
            SpikeTemplate(Y_upper, Y_lower),
            SpikeTemplate(O_upper, O_lower),
            SpikeTemplate(M_upper, M_lower), #This one should have the best cluster
        ]

    def generate(self, n_copies):

        spikes_per_template = [n_copies] * len(self.basis)

        for idx, template in enumerate(self.basis):
            #d = 'gaussian' takes longer but gives better plot
            template.generate(spikes_per_template[idx],d='uniform')
            self.spikes.append(template.spikes)

        return [spikes for neuron in self.spikes for spikes in neuron]

    def show(self):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
        fig.suptitle('Ground truth')

        # Plot all spikes 
        for neuron in self.spikes:
            for spike in neuron:
                ax1.plot(range(40), spike, 'b')

        #Plot sorted spikes
        colors = ['deepskyblue','darkblue','green','yellow','orange','maroon']
        for idx, neuron in enumerate(self.spikes):
            for spike in neuron:
                ax2.plot(range(40), spike, colors[idx])

        ax1.title.set_text('Unsorted')
        ax2.title.set_text('Sorted')
        plt.show(block=True)

def get_sample_spikes():

    n_copies = 20
    
    sg = SpikeGenerator()

    spikes = sg.generate(n_copies)

    #sg.show()

    return spikes