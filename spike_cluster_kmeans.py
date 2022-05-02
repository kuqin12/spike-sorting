import logging
import numpy as np
import random

from pyspark.sql import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
from math import sqrt

MAX_ITER = 20

class SpikeClusterKMeans(object):

  def __init__(self, spark_cntxt):
    self.sc = spark_cntxt

  # Helper function to calculate euclidean distance
  # Note that this is square as is!!!
  @staticmethod
  def e_dist (point_a, point_b):
    diff = point_a - point_b
    ret = np.sum (diff * diff)
    return ret

  @staticmethod
  def eval_cnt (point, center):
    min_dist = -1
    new_cluster = None
    int_k = len (center)
    for cnt_idx in range (0, int_k):
      measure = SpikeClusterKMeans.e_dist (center[cnt_idx], point)
      if min_dist == -1 or min_dist > measure:
        new_cluster = cnt_idx
        min_dist = measure
    return (new_cluster, (min_dist, point))

  # Implement k-means
  def KMeans(self, waveforms, k=3):

    # for initialization, we randomly select k centroids
    centroids = [None] * k
    for idx in range (k):
      centroids[idx] = random.choice(waveforms)

    std = [0] * k
    dev = [0] * k

    data = self.sc.textFile(waveforms).map(lambda line: np.fromstring (line, sep=' '))

    # 3: for iteration := 1 to MAX ITER do
    for _ in range (0, MAX_ITER + 1):
      # 4: for each point x in the dataset do
      # 5: Cluster of x ← the cluster with the closest centroid to x
      # 6: end for
      data_group = data.map (lambda a: SpikeClusterKMeans.eval_cnt (a, centroids))

      # 10: Calculate the cost for this iteration.
      cost_raw = data_group.mapValues(lambda a: a[0]).reduceByKey (lambda n1, n2: n1 + n2).collect()
      for idx, dist in cost_raw:
        dev[idx] = dist

      # 7: for each cluster P do
      # 8: Centroid of P ← the mean of all the data points assigned to P
      # 9: end for
      mean = sorted(data_group.mapValues(lambda a: (a[1], 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).collect())
      for idx, (total, count) in mean:
        centroids[idx] = total / count
        dev[idx] = dev[idx] / count

    for idx in range(k):
      std[idx] = sqrt(dev[idx])

    # 11: end for
    return (centroids, std)
