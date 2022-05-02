
import logging
import numpy as np

from pyspark.sql import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
from math import sqrt

MAX_ITER = 20

class SpikeFeatureExtractPCA(object):

  def __init__(self, spark_cntxt):
    self.sc = spark_cntxt

  def PCA(self, origin_data, k=10):

    row = len(origin_data)
    col = len(origin_data[0])
    logging.info ("Row: %d, Column: %d" % (row, col))

    sigma = np.zeros ([col, col])

    # This section needs improvement
    for idx in range (0, row):
      if (idx % 100) == 0:
        logging.info ("Progress: %.3f%%" % (idx*100/row))
      sigma += np.outer(origin_data[idx],origin_data[idx])

    sigma = sigma / row

    logging.info ("Start to solve eigen values...")
    eig_val, eig_vec = np.linalg.eig (sigma)
    logging.info ("Solve done!")

    recon_wave = [None] * row
    for idx in range (0, row):
      ori_wave = origin_data[idx]
      recon_vec = eig_vec[:, :k]
      recon_wave[idx] = ori_wave.dot(recon_vec.dot(recon_vec.T)).real

    return recon_wave
