import logging
import numpy as np
import random

from pyspark.sql import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
from math import sqrt
from spike_cluster_type import SpikeClustering

class SpikeClusterKMeans(SpikeClustering):

  def __init__(self, spark):
    self.sp = spark

  # Helper function to calculate euclidean distance
  # Note that this is square as is!!!
  @staticmethod
  def e_dist (point_a, point_b):
    diff = point_a - point_b
    ret = diff * diff
    return ret

  @staticmethod
  def eval_cnt (point, center):
    min_dist = -1
    new_cluster = None
    int_k = len (center)
    for cnt_idx in range (0, int_k):
      ret = SpikeClusterKMeans.e_dist (center[cnt_idx], point)
      measure = sum (ret)
      if min_dist == -1 or min_dist > measure:
        new_cluster = cnt_idx
        min_dist = measure
    return (new_cluster, ret, point)

  @staticmethod
  def update_cent (entry, center):
    idx = entry[0]
    total = entry[1][0]
    count = entry[1][1]
    center[idx] = total/count
    return entry

  # Evaluate clustering by computing Within Set Sum of Squared Errors
  @staticmethod
  def error(point, centers):
    min_dist = -1
    new_cluster = None
    int_k = len (centers)
    for cnt_idx in range (0, int_k):
      ret = SpikeClusterKMeans.e_dist (centers[cnt_idx], point)
      measure = sum (ret)
      if min_dist == -1 or min_dist > measure:
        new_cluster = cnt_idx
        min_dist = measure
    return (new_cluster, (ret, 1))

  # Implement k-means
  def Cluster(self, waveforms, k=3, max_iter=20):

    # for initialization, we randomly select k centroids
    centroids = [None] * k
    for idx in range (k):
      centroids[idx] = random.choice(waveforms)

    std = [0] * k

    logging.debug ("Initialize RDD")
    data = self.sp.sparkContext.parallelize(waveforms)
    logging.debug ("Keamns entry")

    # 3: for iteration := 1 to MAX ITER do
    for round in range (0, max_iter + 1):
      clusters = [ [] for _ in range(k) ]
      # 4: for each point x in the dataset do
      # 5: Cluster of x ← the cluster with the closest centroid to x
      # 6: end for
      logging.info ("Round %d, test point 1" % round)
      for idx in range (len(waveforms)):
        res = SpikeClusterKMeans.eval_cnt (waveforms[idx], centroids)
        clusters[res[0]].append ((idx, res[1], res[2]))

      # 7: for each cluster P do
      # 8: Centroid of P ← the mean of all the data points assigned to P
      # 9: end for
      logging.info ("Round %d, test point 2" % round)
      for idx in range(k):
        cluster = clusters[idx]
        if len (cluster) > 0:
          centroids[idx] = sum(point for _, _, point in cluster) / len (cluster)
      logging.info ("Round %d, test point 3" % round)

      logging.debug ("Round %d complete" % round)

    # 7: for each cluster P do
    # 8: Centroid of P ← the mean of all the data points assigned to P
    # 9: end for
    for idx in range(k):
      clusters[idx] = [i for i, _, _ in clusters[idx]]

    # 11: end for
    return clusters
