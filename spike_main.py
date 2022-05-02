import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import *

import spike_filter_detect as sp_fd

from spike_cluster_kmeans import SpikeClusterKMeans
from spike_fe import SpikeFeatureExtractPCA
from spike_cluster_kmeans_mllib import SpikeClusterKMeans_MLLib
from spike_fe_mllib import SpikeFeatureExtractPCA_MLLib

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
  sc.setLogLevel("warn")

  # This is Kun's home brew implementation
  skm = SpikeClusterKMeans (sc)
  sfe = SpikeFeatureExtractPCA (sc)

  # # This is purely from spark MLLib
  # skm = SpikeClusterKMeans_MLLib (sc)
  # sfe = SpikeFeatureExtractPCA_MLLib (sc)

  with open (Paths.InputFile, 'rb') as input:
    # Detect spike, based on 
    raw_data = input.read (Paths.Interval * SAMPLE_FREQ * Paths.Channels)

    a = np.frombuffer(raw_data, dtype=np.uint16)

    # TODO: Need to parallelize this
    for chn in range (Paths.Channels):
      sliced_data = raw_data[chn::Paths.Channels]

      spike_data = sp_fd.filter_data(sliced_data, low=500, high=9000, sf=SAMPLE_FREQ)
      plt.plot (wave_form)

      _, wave_form = sp_fd.get_spikes(spike_data, spike_window=50, tf=8, offset=20)
      plt.plot (wave_form)

      # Now we are ready to cook. Start from feature extraction
      extracted_wave = sfe.PCA (wave_form)
      plt.plot (wave_form)

      cluster_means, cluster_std = skm.KMeans (extracted_wave, k=3)
      for idx in range (3):
        plt.plot (cluster_means[idx], label = "line %d" % idx)
        plt.fill_between (cluster_means[idx]-cluster_std[idx], cluster_means[idx]+cluster_std[idx])

  return 0

if __name__ == '__main__':
  ret = main ()
  exit (ret)
