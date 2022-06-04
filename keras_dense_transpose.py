from keras.layers import Layer
from keras import activations
import tensorflow as tf

class DenseTranspose(Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = activations.get(activation)
        super().__init__(**kwargs)
    
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name='bias', shape=[self.dense.input_shape[-1]], initializer='zeros')
        super().build(batch_input_shape)
    
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)
