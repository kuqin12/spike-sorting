import argparse
import logging
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import alphas
from pyspark.sql import *
from scipy.stats import median_abs_deviation

import spike_filter_detect as sp_fd

from spike_fe import SpikeFeatureExtractPCA
from spike_fe_mllib import SpikeFeatureExtractPCA_MLLib

from spike_cluster_kmeans import SpikeClusterKMeans
from spike_cluster_kmeans_mllib import SpikeClusterKMeans_MLLib
from spike_cluster_kmeans_skl import SpikeClusterKmeans_SKL
from spike_cluster_gmm_skl import SpikeClusterGMM_SKL
from spike_cluster_gmm import SpikeClusterGMM

from spike_svm import SpikeSVMClassifier

path_root = os.path.dirname(__file__)
sys.path.append(path_root)

SAMPLE_FREQ       = 30000

VICINITY_CHANNEL      = 8
SIMILARITY_THRESHOLD  = 3
# MINIMAL_CLUSTER_PORT  = 0.005

def FEFactory(InputString, context=None):
  if(InputString.lower() == "pca"):
    return SpikeFeatureExtractPCA (context)

  if(InputString.lower() == "mlpca"):
    return SpikeFeatureExtractPCA_MLLib (context)

  raise Exception("Unsupported Feature Extration String %s" % InputString)

def ClusterFactory(InputString, context=None):
  if(InputString.lower() == "mlkm"):
    return SpikeClusterKMeans_MLLib(context)

  if(InputString.lower() == "km"):
    return SpikeClusterKMeans(context)

  if(InputString.lower() == "sklgmm"):
    return SpikeClusterGMM_SKL(context)

  if(InputString.lower() == "sklkm"):
    return SpikeClusterKmeans_SKL(context)

  if(InputString.lower() == "gmm"):
    return SpikeClusterGMM(context)

  raise Exception("Unsupported Clustering String %s" % InputString)

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
        help = '''Number of channels contained in data file'''
        )
    parser.add_argument (
        '-fe', '--FeatureExtraction', dest = 'FeatureExtraction', type=str, default='pca',
        help = '''Feature extraction method used for decomposition. Currently supported are 'pca' and 'mlpca'.'''
        )
    parser.add_argument (
        '-cls', '--Cluster', dest = 'ClusterMethod', type=str, default='km',
        help = '''Clustering method used for this sorting. Currently supported are 'km', 'mlkm', 'sklkm', 'sklgmm' and 'gmm'.'''
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

def cluster_distance (clusters_m, clusters_n):
  # The maths is from https://doi.org/10.7554/eLife.34518
  alpha_m = np.mean(clusters_m, axis=0)
  alpha_n = np.mean(clusters_n, axis=0)
  dist = alpha_m - alpha_n
  # print (dist.shape)
  # print (clusters_m.shape)
  # print (clusters_n.shape)
  gamma_m = median_abs_deviation(np.dot(clusters_m, dist))
  gamma_n = median_abs_deviation(np.dot(clusters_n, -dist))
  zeta_m_n =  np.linalg.norm(dist) / np.sqrt(gamma_m * gamma_m + gamma_n * gamma_n)
  # print (zeta_m_n)
  return zeta_m_n

def main ():

  # Parse required paths passed from cmd line arguments
  Paths = path_parse()

  # create the Spark Session
  # spark = SparkSession.builder.getOrCreate()
  spark = None

  # # create the Spark Context
  # sc = spark.sparkContext

  # # Mute the informational level logs
  # sc.setLogLevel("WARN")

  # This is Kun's home brew implementation
  fe_model = FEFactory (Paths.FeatureExtraction, spark)
  cluster_model = ClusterFactory (Paths.ClusterMethod, spark)
  svm_classifier = SpikeSVMClassifier(spark)
  start_time = 0

  with open (Paths.InputFile, 'rb') as input:
    # Detect spike, the 2 factor is for the 16bit integer
    raw_data = input.read (Paths.Interval * SAMPLE_FREQ * Paths.Channels * 2)

    raw_data_16 = np.frombuffer(raw_data, dtype=np.int16)
    timestamp = [(item / SAMPLE_FREQ) for item in range(start_time, start_time + SAMPLE_FREQ * Paths.Interval)]

    # TODO: Need to parallelize this
    summary_list = []
    for chn in range (Paths.Channels):
      sliced_data = raw_data_16[chn::Paths.Channels]

      spike_data = sp_fd.filter_data(sliced_data, low=300, high=6000, sf=SAMPLE_FREQ)
      # plt.plot(timestamp, spike_data)
      # plt.suptitle('Channel %d' % (chn + 1))
      # plt.show ()

      wave_form = sp_fd.get_spikes(spike_data, spike_window=50, tf=8, offset=20, max_thresh=1000)
      if len(wave_form) == 0:
        # no spike here, bail this channel for this interval
        logging.debug ("no spike here, bail this channel for this interval %d." % (chn + 1))
        summary_list.append (None)
        continue

      # Now we are ready to cook. Start from feature extraction
      logging.debug ("Start to process %d waveforms with PCA." % len(wave_form))
      extracted_wave = fe_model.FE (wave_form)

      logging.debug ("Done processing PCA!!!")

      if len(extracted_wave) <= 3:
        # Less than cluster numbers, bail fast for this interval
        logging.debug ("Less than cluster numbers, bail fast for this interval %d." % (chn + 1))
        summary = [None] * len(extracted_wave)
        for idx in range (len(extracted_wave)):
          # c_sum = sum (extracted_wave[idx])
          # c_sum_sq = sum (np.square(extracted_wave[idx]))
          summary[idx] = (extracted_wave[[idx]])
        print (summary)
        summary_list.append (summary)
        continue

      # start = time.time()
      clusters = cluster_model.Cluster (extracted_wave, k=3)
      # end = time.time()
      # logging.debug("The time of execution of above step is : %f" % (end-start))
      logging.debug ("Done clustering!!!")

      # save the clusters by summary, so that they can still be in memory
      summary = [None] * 3
      for idx in range (3):
        cluster = clusters[idx]
        logging.debug (cluster)
        # c_sum = sum (extracted_wave[cluster])
        # c_sum_sq = sum (np.square(extracted_wave[cluster]))
        summary[idx] = (extracted_wave[cluster])

      summary_list.append (summary)

    # Now that we have the list of labeled clusters, now we need to potentially merge the clusters in vicinity
    logging.critical ("Starting to merge cross channel data!!!")
    final_clusters = []
    total_spikes = 0
    for channel in range(VICINITY_CHANNEL, len(summary_list)):
      # Here we process 4 overlapping tetrodes (each has 4 channels) at a time
      channel_summary = summary_list[channel]
      # when the channel did not collect any spikes
      if channel_summary is None:
        continue

      for cluster_summary in channel_summary:
        if cluster_summary is None:
          continue

        total_spikes += len(cluster_summary)
        length = len(final_clusters)
        if length == 0:
          # create one cluster if it is empty
          final_clusters.append ((cluster_summary, channel, channel))
        else:
          # search back to see how many final clusters are mergable with this one
          index = length
          merge_idx = None
          min_cl_dist = -1
          while index > 0:
            (cl_sum, s_chn, _) = final_clusters[index - 1]
            # making sure that the span of merged clusters will not exceed the range of 4 tetrodes
            if channel - s_chn <= VICINITY_CHANNEL:
              cl_dist = cluster_distance (cl_sum, cluster_summary)
              if cl_dist < SIMILARITY_THRESHOLD:
                # There is definitely going to be a merge
                if merge_idx is None or min_cl_dist > cl_dist:
                  merge_idx = index - 1
                  min_cl_dist = cl_dist
            else:
              # Did not find anything to merge
              break
            index -= 1

          # Update the cluster summary after all calculations
          if merge_idx is not None:
            final_clusters[merge_idx] = (np.vstack((cluster_summary, cl_sum)), s_chn, channel)
          else:
            final_clusters.append((cluster_summary, channel, channel))
    logging.critical ("Done merging spikes. Total %d spikes and %d clusters!!!" % (total_spikes, len(final_clusters)))

    # Lastly, classify the results with SVM
    labels = [None] * total_spikes
    all_waves = None
    index = 0
    for idx, each in enumerate(final_clusters):
      if all_waves is None:
        all_waves = each[0]
      else:
        all_waves = np.vstack((all_waves, each[0]))
      for _ in range(len(each[0])):
        # labels[index] = idx
        index += 1
    if index != total_spikes:
      raise Exception ("Something is off %d %d" % (index, total_spikes))

    svm_classifier.Fit (data=all_waves, label=labels)

  return 0

if __name__ == '__main__':
  ret = main ()
  exit (ret)
