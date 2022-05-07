import logging
import numpy as np
import random

from pyspark.sql import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
from math import sqrt
from sklearn.cluster import KMeans
from spike_cluster_type import SpikeClustering

class SpikeClusterKmeans_SKL(SpikeClustering):

  def __init__(self, spark):
    self.sp = spark

  # Fit into Gaussian Mixture Model
  def Cluster(self, waveforms, k=3, max_iter=20):

    km = KMeans(n_clusters=k, max_iter=max_iter)
    km.fit(waveforms)
    labels = km.labels_

    clusters = [ [] for _ in range(k) ]

    for idx in range(k):
      clusters[idx] = [i for i in range(len(labels)) if labels[i] == idx]

    return clusters
