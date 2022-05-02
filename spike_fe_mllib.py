from pyspark.ml.feature import PCA
from pyspark.mllib.linalg import Vectors

class SpikeFeatureExtractPCA_MLLib(object):

  def __init__(self, spark_cntxt):
    self.sc = spark_cntxt

  def PCA(self, origin_data, k=10):
    pca = PCA(k)
    model = pca.fit(origin_data)

    result = model.transform(origin_data)

    return result
