import logging
import numpy as np
import matplotlib.pyplot as plt

from pyspark.sql import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors
from math import sqrt

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

class SpikeClusterKMeans_MLLib(object):

  def __init__(self, spark):
    self.sp = spark

  # Evaluate clustering by computing Within Set Sum of Squared Errors
  @staticmethod
  def error(point, model, centers):
    center = centers[model.predict(point)]
    return (model.predict(point), (sum([x**2 for x in (point - center)]), 1))

  # Implement k-means
  def KMeans(self, waveforms, k=3, max_iter=20):
    list_data = [(Vectors.dense(form.tolist()),) for form in waveforms]

    # Loads data.
    dataset = self.sp.createDataFrame (list_data, ['features'])

    # Trains a k-means model.
    kmeans = KMeans().setK(k).setSeed(1).setMaxIter(max_iter)
    model = kmeans.fit(dataset, )

    std = [(0,0)] * k
    # Shows the result.
    centers = model.clusterCenters()

    a = dataset.collect()

    clusters = [ [] for _ in range(k) ]
    for idx in range(len(a)):
      pre = model.predict(a[idx].features)
      clusters[pre].append (idx)

    return clusters
