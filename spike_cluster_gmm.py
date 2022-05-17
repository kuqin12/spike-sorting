from cmath import pi
import logging
import numpy as np
import random

from pyspark.sql import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
from math import gamma, sqrt
from spike_cluster_type import SpikeClustering

class SpikeClusterGMM(SpikeClustering):

  def __init__(self, spark):
    self.sp = spark

  # Helper function to calculate euclidean distance
  # Note that this is square as is!!!
  @staticmethod
  def e_gamma (point, mu_i, sigma_i):
    mid = np.matmul ((point - mu_i).T, np.linalg.inv(sigma_i))
    order = -np.matmul (mid, (point - mu_i)) / 2
    det = np.linalg.det (sigma_i)
    ret = (e ^ order) / np.sqrt ((2*np.pi)^k * det)
    return ret

  @staticmethod
  def eval_cnt (point, mu, pi, sigma):
    max_prob = 0
    new_cluster = None
    int_k = len (pi)
    gamma = np.array([0] * int_k)
    for cnt_idx in range (0, int_k):
      ret = SpikeClusterGMM.e_gamma (point, mu[cnt_idx], sigma[cnt_idx])
      if max_prob < ret:
        new_cluster = cnt_idx
        max_prob = ret
      gamma[cnt_idx] = ret * pi[cnt_idx]
    gamma = gamma / sum (gamma)
    return (new_cluster, point, gamma)

  @staticmethod
  def update_cent (entry, center):
    idx = entry[0]
    total = entry[1][0]
    count = entry[1][1]
    center[idx] = total/count
    return entry

  # Implement GMM
  def Cluster(self, waveforms, k=3, max_iter=20):

    # for initialization, we randomly select k vectors
    N = len(waveforms)
    pi = [1/N] * k
    mu = [None] * k
    sigma = [None] * k
    for idx in range (k):
      mu[idx] = random.choice(waveforms)
      sigma[idx] = np.cov (waveforms.T)

    # Iterate towards the convergence
    for round in range (0, max_iter + 1):
      clusters = [ [] for _ in range(k) ]
      # E-Step
      for idx in range (len(waveforms)):
        # Assign to the cluster with the largest value of gamma
        res = SpikeClusterGMM.eval_cnt (waveforms[idx], mu, pi, sigma)
        clusters[res[0]].append ((idx, res[1], res[2]))

      # M-Step
      for idx in range(k):
        sum_d = 0
        sum_s = np.array ([[0] * len(waveforms[0])] * len(waveforms[0]))
        sum_n = 0
        cluster = clusters[idx]
        # Fraction of number of points
        pi[idx] = len (cluster) / N
        for idx_2 in range(k):
          sum_d += sum(point * gamma[idx] for _, point, gamma in clusters[idx_2])
          sum_s += sum(point * point.T * gamma[idx] for _, point, gamma in clusters[idx_2])
          sum_n += sum(point * gamma[idx] for _, point, gamma in clusters[idx_2])
        # Update means based on the points in this cluster
        mu[idx] = sum_d / sum_n
        # Update covariance based on points in this cluster
        sigma[idx] = sum_s / sum_n

    # Collect the result from the last iteration
    for idx in range(k):
      clusters[idx] = [i for i, _, _ in clusters[idx]]

    # 11: end for
    return clusters
