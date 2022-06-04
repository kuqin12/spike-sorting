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

  @staticmethod
  def far_init(waveforms, k):

    # Randomly pick the first centroid
    centroids = [random.choice(waveforms)]

    # from https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
    for _ in range(k-1):
        dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in waveforms])
        probs = dist_sq/dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()

        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break

        centroids.append(waveforms[i])

    return centroids

  # Implement k-means
  def Cluster(self, waveforms, k=3, max_iter=20, init='kpp'):

    if init == 'kpp':
      # for initialization, we randomly select k centroids
      centroids = SpikeClusterKMeans.far_init(waveforms=waveforms, k=k)
    else:
      centroids = [None] * k
      for idx in range (k):
        centroids[idx] = random.choice(waveforms)

    # 3: for iteration := 1 to MAX ITER do
    for _ in range (0, max_iter + 1):
      clusters = [ [] for _ in range(k) ]
      # 4: for each point x in the dataset do
      # 5: Cluster of x ← the cluster with the closest centroid to x
      # 6: end for
      for idx in range (len(waveforms)):
        res = SpikeClusterKMeans.eval_cnt (waveforms[idx], centroids)
        clusters[res[0]].append ((idx, res[1], res[2]))

      # 7: for each cluster P do
      # 8: Centroid of P ← the mean of all the data points assigned to P
      # 9: end for
      for idx in range(k):
        cluster = clusters[idx]
        if len (cluster) > 0:
          centroids[idx] = sum(point for _, _, point in cluster) / len (cluster)

    # 7: for each cluster P do
    # 8: Centroid of P ← the mean of all the data points assigned to P
    # 9: end for
    for idx in range(k):
      clusters[idx] = [i for i, _, _ in clusters[idx]]

    # 11: end for
    return clusters
