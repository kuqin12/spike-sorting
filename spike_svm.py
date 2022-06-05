import argparse
import logging
import os
import math
import numpy as np
import time

class SpikeSVMClassifier (object):

  def __init__(self, spark, GD="MBGD"):
    self.sp = spark
    self.svms = {}
    self.GD = GD
    if GD == "BGD":
      self.eta = 0.0000003
      self.epsilon = 0.25
      self.converge = SpikeSVMClassifier.BGD_Conv
    elif GD == "SGD":
      self.eta = 0.0001
      self.epsilon = 0.001
      self.converge = SpikeSVMClassifier.SGD_Conv
    elif GD == "MBGD":
      self.eta = 0.00001
      self.epsilon = 0.01
      self.converge = SpikeSVMClassifier.MBGD_Conv
    else:
      raise Exception ("Unsupported convergence condition specified.")

  @staticmethod
  def BGD_Conv (loss_arr, delta_loss_arr, epsilon):
    old_loss = loss_arr [-2]
    new_loss = loss_arr [-1]

    delta_perc_loss = abs (old_loss - new_loss) / old_loss * 100
    # print (delta_perc_loss)
    if delta_perc_loss < epsilon:
      return True
    else:
      return False

  @staticmethod
  def SGD_Conv (loss_arr, delta_loss_arr, epsilon):
    old_loss = loss_arr [-2]
    new_loss = loss_arr [-1]

    delta_perc_loss = abs (old_loss - new_loss) / old_loss * 100

    delta_loss = 0.5 * delta_loss_arr[-1] + 0.5 * delta_perc_loss
    delta_loss_arr.append (delta_loss)

    # print (delta_loss)
    if delta_loss < epsilon:
      return True
    else:
      return False

  @staticmethod
  def MBGD_Conv (loss_arr, delta_loss_arr, epsilon):
    return SpikeSVMClassifier.SGD_Conv (loss_arr, delta_loss_arr, epsilon)

  @staticmethod
  def CalculateLoss (w, b, p_features, p_target, C):
    loss = 0.5 * np.dot(w ,w)
    n = len (p_features)
    for i in range(n):
      e_loss = p_target[i] * (np.dot(w, p_features[i]) + b)
      loss += (C * max(0, 1 - e_loss))

    return loss

  @staticmethod
  def unison_shuffled_copies(a, b):
      assert len(a) == len(b)
      p = np.random.permutation(len(a))
      return a[p], b[p]

  def SVM (self, features, target):
    # print ("Start of SVM - %s" % self.GD)
    # start_time = time.time()

    n = len(features)
    d = len(features[0])
    if self.GD == 'BGD':
      p_features = features
      p_target = target
      self.beta = n
    elif self.GD == 'SGD':
      p_features, p_target = SpikeSVMClassifier.unison_shuffled_copies (features, target)
      self.beta = 1
    elif self.GD == 'MBGD':
      p_features, p_target = SpikeSVMClassifier.unison_shuffled_copies (features, target)
      self.beta = 20
    else:
      raise Exception ("Unexpected gradient descent - %s!!!" % self.GD)

    k = 0
    w = [0] * d
    b = 0
    C = 100
    ret_arr = []
    loss = SpikeSVMClassifier.CalculateLoss (w, b, p_features, p_target, C)
    ret_arr.append (loss)
    # print (loss)
    delta_loss = [0]
    t = 1
    while True:
      # cache the old value
      w_old = w.copy()
      b_old = b
      # print ("Iteration %d" % t)
      e_loss = {}
      p_b = 0
      start = self.beta * k
      end = min (self.beta * k + self.beta - 1, n - 1)
      for i in range(start, end + 1):
        # Within the range of B
        e_loss[i] = p_target[i] * (np.dot(w_old, p_features[i]) + b_old)
        # For optimization, sum up the result of equation 2
        p_b += 0 if e_loss[i] >= 1 else (-p_target[i])

      for j in range (d):
        p_wj = 0
        for i in e_loss:
          # Sum up the result of equation 1
          p_wj += 0 if e_loss[i] >= 1 else (-p_target[i] * p_features[i][j])

        w[j] = w_old[j] - self.eta * (w_old[j] + C * p_wj)

      b = b_old - self.eta * C * p_b
      k = (k + 1) % math.ceil(n/self.beta)

      loss = SpikeSVMClassifier.CalculateLoss (w, b, p_features, p_target, C)
      ret_arr.append (loss)
      if self.converge (ret_arr, delta_loss, self.epsilon):
        break

      if t % 50 == 0:
        logging.debug ("Loss for %d iteration %f" % (t, loss))

      t += 1

    # done_time = time.time()
    # elapsed = done_time - start_time
    # print("Time elapsed %ds after %d iterations" % (elapsed, t))

    return w, b

  def Fit(self, data, label):
    # The binary classification is "trivial" as noted above.
    # This one we need to do a one-to-rest multiclass SVM
    unique_label = np.unique(label)
    for each in unique_label:
      # specifically binarize the label
      temp_label = np.array([1 if i == each else -1 for i in label])
      logging.critical ("SVM classification for %d" % each)
      w_each, b_each = self.SVM(data, temp_label)
      logging.critical ("SVM classification done!!!")
      self.svms [each] = (w_each, b_each)

  def Predict(self, sample):
    best_predict = None
    max_predict = None
    for each in self.svms:
      w, b = self.svms[each]
      predict = np.dot(sample, w) + b
      if max_predict == None or predict > max_predict:
        best_predict = each
        max_predict = predict

    # if max_predict < 0:
    #   # Something is off.. this prediction is the best but still may not be accurate
    #   # Do something...

    return best_predict
