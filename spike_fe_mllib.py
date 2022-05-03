from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
import numpy as np
import logging

class SpikeFeatureExtractPCA_MLLib(object):

  def __init__(self, spark):
    self.sp = spark

  def PCA(self, origin_data, k=10):
    list_data = [(Vectors.dense(form.tolist()),) for form in origin_data]

    data = self.sp.createDataFrame (list_data, ['np_waveforms'])

    pca = PCA(k=k, inputCol="np_waveforms", outputCol="pcawaveforms")
    model = pca.fit(data)

    recon = model.transform(data)
    out = recon.select('pcawaveforms').rdd.map(lambda r: r[0]).collect()

    result = [np.array(row) for row in out]

    return result
