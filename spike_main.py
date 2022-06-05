import argparse
import logging
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from pyparsing import alphas
from pyspark.sql import *
import matplotlib.cm as cm

# from generate_data import get_sample_spikes
from repro_DCAE_data import get_sample_spikes
import spike_filter_detect as sp_fd

from spike_fe import SpikeFeatureExtractPCA
from spike_fe_mllib import SpikeFeatureExtractPCA_MLLib

from spike_cluster_kmeans import SpikeClusterKMeans
from spike_cluster_kmeans_mllib import SpikeClusterKMeans_MLLib
from spike_cluster_kmeans_skl import SpikeClusterKmeans_SKL
from spike_cluster_polisher import merge_clusters, filter_clusters
from spike_cluster_gmm_skl import SpikeClusterGMM_SKL
from spike_cluster_gmm import SpikeClusterGMM
from tsne import scatter

from spike_svm import SpikeSVMClassifier
from deep_contractive_autoencoder import get_model, get_encoder

path_root = os.path.dirname(__file__)
sys.path.append(path_root)

SAMPLE_FREQ = 30000
MAX_CLUSTER_PER_CHN   = 10
MIN_CLUSTER_PER_CHN   = 3

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

def main ():

  # Parse required paths passed from cmd line arguments
  Paths = path_parse()

  # # create the Spark Session
  # spark = SparkSession.builder.getOrCreate()
  spark = None

  # # create the Spark Context
  # sc = spark.sparkContext

  # # Mute the informational level logs
  # sc.setLogLevel("WARN")

  # This is Kun's home brew implementation
  fe_model = FEFactory (Paths.FeatureExtraction, spark)
  cluster_model = ClusterFactory (Paths.ClusterMethod, spark)
  svm_classifier = SpikeSVMClassifier(spark, GD="BGD")
  start_time = 0

  with open (Paths.InputFile, 'rb') as input:
    # Detect spike, the 2 factor is for the 16bit integer
    raw_data = input.read (Paths.Interval * SAMPLE_FREQ * Paths.Channels * 2)

    #raw_data_16 = np.frombuffer(raw_data, dtype=np.int16)
    timestamp = [(item / SAMPLE_FREQ) for item in range(start_time, start_time + SAMPLE_FREQ * Paths.Interval)]

    # TODO: Need to parallelize this
    summary_list = []
    total_spikes = 0
    for chn in range (Paths.Channels):
      logging.critical ("Progress: %.1f%%..." % (chn/Paths.Channels*100))
      #sliced_data = raw_data_16[chn::Paths.Channels]

      #spike_data = sp_fd.filter_data(sliced_data, low=300, high=6000, sf=SAMPLE_FREQ)
      # plt.plot(timestamp, spike_data)
      # plt.suptitle('Channel %d' % (chn + 1))
      # plt.show ()

      # wave_form = sp_fd.get_spikes(spike_data, spike_window=50, tf=8, offset=20, max_thresh=1000)
      # if len(wave_form) == 0:
      #   # no spike here, bail this channel for this interval
      #   logging.debug ("no spike here, bail this channel for this interval %d." % (chn + 1))
      #   continue

      # if (len(wave_form) == 1):
      #   # TODO: This is not right, but different reduction should remove it
      #   continue

      # # TODO: This should not be needed with curated data
      # min_vals = np.min(wave_form, axis=1)
      # wave_form = wave_form + min_vals[:, None]
      # max_val = np.max(wave_form, axis=1)
      # wave_form = wave_form / max_val[:, None]

      wave_form = []
      for _ in range(30):
        wave_form += get_sample_spikes ()
      total_spikes = len(wave_form)
      # Now we are ready to cook. Start from feature extraction
      ########
      # PCA
      ########
      # logging.debug ("Start to process %d waveforms with PCA." % len(wave_form))
      # extracted_wave = fe_model.FE (wave_form)
      # logging.debug ("Done processing PCA!!!")

      ########
      # DCAE
      ########
      logging.debug ("Start to process %d waveforms with DCAE." % len(wave_form))
      input_waves = np.array(wave_form)
      wave_dim = input_waves.shape[1]
      hidden_layers = [25, 10]
      autoencoder = get_model(input_waves, wave_dim, hidden_layers, epochs=500, batch_size=10)
      encoder = get_encoder(autoencoder, wave_dim, len(hidden_layers))
      extracted_wave = encoder.predict(input_waves)
      logging.debug ("Done processing DCAE!!!")

      if len(extracted_wave) <= MIN_CLUSTER_PER_CHN:
        # Less than cluster numbers, bail fast for this interval
        logging.debug ("Less than cluster numbers, bail fast for this interval %d." % (chn + 1))
        for idx in range (len(extracted_wave)):
          # c_sum = sum (extracted_wave[idx])
          # c_sum_sq = sum (np.square(extracted_wave[idx]))
          summary_list.append ((extracted_wave[[idx]], chn, chn, np.array(wave_form)[[idx]]))
        continue

      n_cluster = int(ceil(len(extracted_wave) / 100))
      if n_cluster < MIN_CLUSTER_PER_CHN:
        n_cluster = MIN_CLUSTER_PER_CHN
      elif n_cluster > MAX_CLUSTER_PER_CHN:
        n_cluster = MAX_CLUSTER_PER_CHN
      # start = time.time()
      clusters = cluster_model.Cluster (extracted_wave, k=n_cluster)
      # end = time.time()
      # logging.debug("The time of execution of above step is : %f" % (end-start))
      logging.debug ("Done clustering!!!")

      # Format the clusters
      for idx in range (n_cluster):
        cluster = clusters[idx]
        logging.debug (cluster)
        if len(cluster) == 0:
          continue
        # c_sum = sum (extracted_wave[cluster])
        # c_sum_sq = sum (np.square(extracted_wave[cluster]))
        summary_list.append ((extracted_wave[cluster], chn, chn, np.array(wave_form)[cluster]))

    # Now that we have the list of labeled clusters, now we need to potentially merge the clusters in vicinity
    logging.critical ("Starting to merge cross channel, total %d spikes from %d clusters!!!" % (total_spikes, len(summary_list)))
    final_clusters = merge_clusters (summary_list, similarity=2, max_iter=0)
    logging.critical ("Done merging spikes. Total %d clusters!!!" % (len(final_clusters)))

    # Lastly, classify the results with SVM
    encoded_waves, all_waves, labels, n_thrown_spikes = filter_clusters(final_clusters, total_spikes)
    logging.critical ("Done filtering. Ended up %d clusters and threw away %d spikes!!!" % (len(np.unique(labels)), n_thrown_spikes))

    logging.critical ("Started SVM classification!!!")
    svm_classifier.Fit (data=encoded_waves, label=labels)
    logging.critical ("Done SVM classification!!!")

    scatter(all_waves, labels)

    # Consume trained SVMs
    new_spikes = get_sample_spikes ()
    new_encoded_spikes = encoder.predict(np.array(new_spikes))
    new_labels = []
    for each in new_encoded_spikes:
      res = svm_classifier.Predict (each)
      new_labels.append(res)

    cmap = cm.get_cmap('Set1')
    for l in np.unique(new_labels):
      i = np.where(new_labels == l)
      labeled_waves = np.array(new_spikes)[i]
      for each in labeled_waves:
        plt.plot(each, c = cmap(l))
    plt.show()

  return 0

if __name__ == '__main__':
  ret = main ()
  exit (ret)
