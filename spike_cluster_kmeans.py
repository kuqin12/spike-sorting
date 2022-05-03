import logging
import numpy as np
import random

from pyspark.sql import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
from math import sqrt

class SpikeClusterKMeans(object):

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
    return (new_cluster, (ret, point))

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
  def KMeans(self, waveforms, k=3, max_iter=20):

    # for initialization, we randomly select k centroids
    centroids = [None] * k
    for idx in range (k):
      centroids[idx] = random.choice(waveforms)

    std = [0] * k
    dev = [0] * k

    logging.debug ("Initialize RDD")
    data = self.sp.sparkContext.parallelize(waveforms)
    logging.debug ("Keamns entry")

    # 3: for iteration := 1 to MAX ITER do
    for round in range (0, max_iter + 1):
      # 4: for each point x in the dataset do
      # 5: Cluster of x ← the cluster with the closest centroid to x
      # 6: end for
      logging.info ("Round %d, test point 1" % round)
      data_group = data.map (lambda a: SpikeClusterKMeans.eval_cnt (a, centroids))

      # 7: for each cluster P do
      # 8: Centroid of P ← the mean of all the data points assigned to P
      # 9: end for
      logging.info ("Round %d, test point 4" % round)
      res = data_group.mapValues(lambda a: (a[1], 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).map (lambda a: SpikeClusterKMeans.update_cent (a, centroids))

      logging.debug ("Round %d complete" % round)

    ret = res.collect()
    std_c = sorted(data.map(lambda a: SpikeClusterKMeans.error(a, centroids)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).mapValues(lambda v: np.sqrt(v[0] / v[1])).collect())

    for idx, std_v in std_c:
      std[idx] = std_v

    # 11: end for
    return (centroids, std)
