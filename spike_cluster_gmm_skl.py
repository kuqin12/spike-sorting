import logging
import numpy as np
import random

from pyspark.sql import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
from math import sqrt
from sklearn import mixture
from spike_cluster_type import SpikeClustering

class SpikeClusterGMM_SKL(SpikeClustering):

  def __init__(self, spark):
    self.sp = spark

  # Fit into Gaussian Mixture Model
  def Cluster(self, waveforms, k=3, max_iter=20):

    gmm = mixture.GaussianMixture(n_components=k, max_iter=max_iter)
    gmm.fit(waveforms)
    labels = gmm.predict(waveforms)

    clusters = [ [] for _ in range(k) ]

    for idx in range(k):
      clusters[idx] = [i for i in range(len(labels)) if labels[i] == idx]

    return clusters
