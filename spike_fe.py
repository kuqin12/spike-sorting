
import logging
import numpy as np

from pyspark.sql import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
from math import sqrt

class SpikeFeatureExtractPCA(object):

  def __init__(self, spark):
    self.sp = spark

  def PCA(self, origin_data, k=10):

    row = len(origin_data)
    col = len(origin_data[0])
    logging.info ("Row: %d, Column: %d" % (row, col))

    means = origin_data - np.mean(origin_data , axis = 0)

    sigma = np.zeros ([col, col])

    # This section needs improvement
    for idx in range (0, row):
      if (idx % 100) == 0:
        logging.info ("Progress: %.3f%%" % (idx*100/row))
      sigma += np.outer(means[idx],means[idx])

    sigma = sigma / row

    logging.info ("Start to solve eigen values...")
    eig_val, eig_vec = np.linalg.eig (sigma)
    logging.info ("Solve done!")

    eig_sub = eig_vec[:, :k]
    recon_wave = np.dot(eig_sub.T, means.T).T.real

    return recon_wave
