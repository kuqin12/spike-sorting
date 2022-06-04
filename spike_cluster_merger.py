import logging
import numpy as np
import random

from pyspark.sql import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
from math import sqrt
from spike_cluster_type import SpikeClustering
from scipy.stats import median_abs_deviation


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

def merge_clusters (summary_list, vicinity_chn=8, similarity=3, max_iter=8):

  rounds = 0
  last_length = 0

  # Stop iteration either it reaches maximal rounds or saturated
  while rounds < max_iter and (len(summary_list) != last_length):

    final_clusters = []
    last_length = len(summary_list)
    for idx in range(len(summary_list)):
      # Here we process 4 overlapping tetrodes (each has 4 channels) at a time
      cluster_summary = summary_list[idx]
      cluster_inst = cluster_summary[0]
      end_channel = cluster_summary[2]

      length = len(final_clusters)
      if length == 0:
        # create one cluster if it is empty
        final_clusters.append (cluster_summary)
        continue

      # search back to see how many final clusters are mergable with this one
      index = length
      merge_idx = None
      min_cl_dist = -1
      while index > 0:
        (cl_sum, s_chn, _) = final_clusters[index - 1]
        # making sure that the span of merged clusters will not exceed the range of 4 tetrodes
        if end_channel - s_chn <= vicinity_chn:
          cl_dist = cluster_distance (cl_sum, cluster_inst)
          if cl_dist < similarity:
            # There is definitely going to be a merge
            if merge_idx is None or min_cl_dist > cl_dist:
              merge_idx = index - 1
              min_cl_dist = cl_dist
        else:
          # Did not find anything to merge
          break
        index -= 1

      # Update the clusters after all calculations
      if merge_idx is not None:
        (merge_cl, merge_s_chn, _) = final_clusters[merge_idx]
        final_clusters[merge_idx] = (np.vstack((cluster_inst, merge_cl)), merge_s_chn, end_channel)
      else:
        final_clusters.append(cluster_summary)

    summary_list = final_clusters
    rounds += 1
    logging.debug ("Round %d: %d clusters!!!" % (rounds, len(final_clusters)))

  return final_clusters
