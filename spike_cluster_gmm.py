from cmath import pi
import logging
import numpy as np
import random

from pyspark.sql import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
# from math import gamma, sqrt
from scipy.stats import multivariate_normal
from spike_cluster_type import SpikeClustering

class SpikeClusterGMM(SpikeClustering):

  def __init__(self, spark):
    self.sp = spark

  # Implement GMM
  def Cluster(self, waveforms, k=3, max_iter=20, reg_cov=1e-6):

    # for initialization, we randomly select k vectors
    N = len(waveforms)
    pi = [1/k] * k
    mu = [None] * k
    sigma = [None] * k
    reg_cov = 1e-6*np.identity(len(waveforms[0]))
    for idx in range (k):
      mu[idx] = random.choice(waveforms)
      sigma[idx] = np.zeros((len(waveforms[0]), len(waveforms[0])))
      np.fill_diagonal(sigma[idx], np.cov(waveforms.T).diagonal() + reg_cov)

    # Iterate towards the convergence
    for round in range (0, max_iter + 1):
      prob = np.zeros ((N, k))
      # E-Step
      # Calculate probability belonging a specific cluster
      for cnt_idx in range (0, k):
        cov = sigma[cnt_idx] + reg_cov
        distribution = multivariate_normal (mean=mu[cnt_idx], cov=cov, allow_singular=True)
        prob[:,cnt_idx] = distribution.pdf(waveforms)
      weighted = prob * pi
      sum_weight = weighted.sum(axis=1)[:, np.newaxis]
      gamma = weighted / sum_weight # same shape as prob: N x k

      # M-Step
      for idx in range(k):
        gamma_i = gamma[:, [idx]]
        sum_gamma_i = sum (gamma[:, idx])
        # Fraction of number of points
        pi[idx] = sum_gamma_i / N
        # Update means based on the points in this cluster
        mu[idx] = (waveforms * gamma_i).sum(axis=0) / sum_gamma_i
        # Update covariance based on points probablity in this cluster
        # TODO: KQ: This equation is wrong on the report, the transpose should be on the first x-u
        sigma[idx] = np.dot((gamma_i*(waveforms-sum_gamma_i)).T,(waveforms-sum_gamma_i))/sum_gamma_i + reg_cov

    # Collect the result by looking up the maximal probability
    clusters = [None] * k
    prob = np.zeros ((N, k))
    for cnt_idx in range (k):
      distribution = multivariate_normal (mean=mu[cnt_idx], cov=sigma[cnt_idx], allow_singular=True)
      prob[:,cnt_idx] = distribution.pdf(waveforms)
    max_index = prob.argmax(axis=1)
    for idx in range(N):
      if clusters[max_index[idx]] is None:
        clusters[max_index[idx]] = [idx]
      else:
        clusters[max_index[idx]].append(idx)

    # 11: end for
    return clusters
