import argparse
import logging
import os
import sys
import wave
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import alphas
from pyspark.sql import *

import spike_filter_detect as sp_fd

from spike_cluster_kmeans import SpikeClusterKMeans
from spike_fe import SpikeFeatureExtractPCA
from spike_cluster_kmeans_mllib import SpikeClusterKMeans_MLLib
from spike_fe_mllib import SpikeFeatureExtractPCA_MLLib

path_root = os.path.dirname(__file__)
sys.path.append(path_root)

SAMPLE_FREQ = 30000

# Setup import and argument parser
def path_parse():

    parser = argparse.ArgumentParser()

    parser.add_argument (
        '-i', '--input', dest = 'InputFile', type=str, required=True,
        help = '''Input file of friend lists.'''
        )
    parser.add_argument (
        '-o', '--output', dest = 'OutFile', type=str, default="clusters.txt",
        help = '''Output file for your recommended list. By default it will be rules.txt under current folder'''
        )
    parser.add_argument (
        '-int', '--interval', dest = 'Interval', type=int, default=3,
        help = '''Intervals in seconds to process the data, the bigger, the more accurate'''
        )
    parser.add_argument (
        '-c', '--channels', dest = 'Channels', type=int, default=384,
        help = '''Intervals in seconds to process the data, the bigger, the more accurate'''
        )

    Paths = parser.parse_args()

    if not os.path.isfile (Paths.InputFile):
      raise Exception ("Input file could not be found!")

    dir = os.path.dirname(Paths.OutFile)
    if dir == '':
        dir = os.path.dirname (__file__)

    if not os.path.isdir(dir):
      os.makedirs(dir)

    return Paths

def main ():

  # Parse required paths passed from cmd line arguments
  Paths = path_parse()

  # create the Spark Session
  spark = SparkSession.builder.getOrCreate()

  # create the Spark Context
  sc = spark.sparkContext

  # Mute the informational level logs
  sc.setLogLevel("WARN")

  # This is Kun's home brew implementation
  skm = SpikeClusterKMeans (spark)
  sfe = SpikeFeatureExtractPCA (spark)

  # # This is purely from spark MLLib
  # skm = SpikeClusterKMeans_MLLib (spark)
  # sfe = SpikeFeatureExtractPCA_MLLib (spark)
  start_time = 0

  with open (Paths.InputFile, 'rb') as input:
    # Detect spike, the 2 factor is for the 16bit integer
    raw_data = input.read (Paths.Interval * SAMPLE_FREQ * Paths.Channels * 2)

    raw_data_16 = np.frombuffer(raw_data, dtype=np.int16)
    timestamp = [(item / SAMPLE_FREQ) for item in range(start_time, start_time + SAMPLE_FREQ * Paths.Interval)]

    # TODO: Need to parallelize this
    for chn in range (Paths.Channels):
      sliced_data = raw_data_16[chn::Paths.Channels]

      spike_data = sp_fd.filter_data(sliced_data, low=300, high=6000, sf=SAMPLE_FREQ)
      plt.plot(timestamp, spike_data)
      plt.suptitle('Channel %d' % (chn + 1))
      plt.show ()

      wave_form = sp_fd.get_spikes(spike_data, spike_window=50, tf=8, offset=20, max_thresh=1000)
      if len(wave_form) == 0:
        # no spike here, bail this channel for this interval
        logging.critical ("no spike here, bail this channel for this interval %d." % (chn + 1))
        continue

      # Now we are ready to cook. Start from feature extraction
      logging.critical ("Start to process %d waveforms with PCA." % len(wave_form))
      extracted_wave = sfe.PCA (wave_form)

      logging.critical ("Done processing PCA!!!")

      clusters = skm.KMeans (extracted_wave, k=3)
      for idx in range (3):
        cluster = clusters[idx]
        logging.critical (cluster)

  return 0

if __name__ == '__main__':
  ret = main ()
  exit (ret)
