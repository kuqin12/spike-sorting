import logging
import numpy as np
import matplotlib.pyplot as plt

from pyspark.sql import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
from math import sqrt

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

MAX_ITER = 20

class SpikeClusterKMeans_MLLib(object):

  def __init__(self, spark_cntxt):
    self.sc = spark_cntxt

  # Evaluate clustering by computing Within Set Sum of Squared Errors
  @staticmethod
  def error(point, model, centers):
      center = centers[model.predict(point)]
      return (model.predict(point), (sum([x**2 for x in (point - center)]), 1))

  # Implement k-means
  def KMeans(self, waveforms, k=3):

    # Loads data.
    dataset = self.sc.read.format("libsvm").load(waveforms)

    # Trains a k-means model.
    kmeans = KMeans().setK(k).setSeed(1).setMaxIter(MAX_ITER)
    model = kmeans.fit(dataset)

    std = [0] * k
    # Shows the result.
    centers = model.clusterCenters()
    for center in centers:
      logging.info (center)

    std_c = sorted(dataset.map(lambda point: SpikeClusterKMeans_MLLib.error(point, centers, model)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).mapValues(lambda v: sqrt(v[0] / v[1])).collect())

    for idx, std_v in std_c:
      std[idx] = std_v

    return (centers, std)
