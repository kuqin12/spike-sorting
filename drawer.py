import argparse
import logging
import os
import math
import numpy as np
import time
import matplotlib.pyplot as plt

def draw_err_vs_sample ():

  # plt.figure(figsize=(8, 6), dpi=300)
  sample = np.array ([120, 1200, 2400, 3600, 4800, 6000])
  error = np.array ([44.833, 49.333, 29.333, 20.5, 9.41, 4])

  plt.xlim(0, 6100)
  plt.ylim(0, 55)
  # plt.axvline(x = 25)
  # plt.axhline(y = 15, xmax=25/40)
  # plt.axhline(y = 20, xmin=25/40)
  # plt.axvline(x = 10, ymin=0, ymax=15/40)
  # plt.axvline(x = 5, ymin=15/40)

  # plt.text(4, 7,       'A', fontsize = 20)
  # plt.text(16.5, 7,    'B', fontsize = 20)
  # plt.text(1.5, 26.5,  'C', fontsize = 20)
  # plt.text(16, 26.5,   'D', fontsize = 20)
  # plt.text(31.5, 7,    'E', fontsize = 20)
  # plt.text(31.5, 26.5, 'F', fontsize = 20)
  plt.plot (sample, error, 'o-')

  plt.xlabel("Numer of Samples for Training")
  plt.ylabel("Error (%)")
  plt.suptitle("Prediction Error vs. Training Size")
  plt.show()

def main ():

  draw_err_vs_sample ()
  return 0

if __name__ == '__main__':
  ret = main ()
  exit (ret)
