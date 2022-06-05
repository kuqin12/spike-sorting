from unicodedata import name
from keras.layers import Input, Dense, Dropout
from keras.models import Model
import keras.backend as K

from keras_dense_transpose import DenseTranspose
from keras_dense_tied import DenseTied

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# pass the input data and specify the hidden_layers in the mode
# hidden_layers should be an array of hidden layer dimensions e.g., [layer1_dim, layer2_dim]
def get_model(input_data, input_dim, hidden_layers, lam=1e-4, epochs=1, batch_size=1):
    # Helper:
    # custom contractive loss function
    current_layer_name= None
    def contractive_loss(y_pred, y_true):
        mse = K.mean(K.square(y_true - y_pred), axis=1)
        W = K.variable(value=model.get_layer(current_layer_name).get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = model.get_layer(current_layer_name).output
        dh = h * (1 - h)  # N_batch x N_hidden
        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)
        return mse + contractive
    
    # model each autoencoder layer separately: first encoder1 & decoder1, then encoder2 & decoder2...
    model = None
    num_layers = len(hidden_layers)
    weights_cache = {}  # layer name - weights learned
    for i in range(1, num_layers+1):
        print(f"learning encoder_{i} and decoder_{i} layers")
        input = Input(shape=(input_dim,))
        prev_layer = input
        encoder_layers = {}  # cache the encoder layers to construct DenseTied decoder layers
        # set up the encoder & the following dropout layers
        for curr_layer in range(1, i+1):
            layer_name = f"encoder_{curr_layer}"
            current_layer_name = layer_name  # keep the latest (innermost) encoder layer
            encoder_layer = Dense(hidden_layers[curr_layer-1], activation='sigmoid', name=layer_name)
            # print(f"{layer_name} output dimension: {hidden_layers[curr_layer-1]}")
            encoder_layers[curr_layer] = encoder_layer
            prev_layer = encoder_layer(prev_layer)
            prev_layer = Dropout(0.05)(prev_layer)
        # set up the decoder & the middle dropout layers
        for curr_layer in range(i, 1, -1):
            layer_name = f"decoder_{curr_layer}"
            # print(f"{layer_name} output dimension: {hidden_layers[curr_layer-2]}")
            # decoder_layer = DenseTied(hidden_layers[curr_layer-2], activation='sigmoid', name=layer_name, tied_to=encoder_layers[curr_layer])
            decoder_layer = DenseTranspose(dense=encoder_layers[curr_layer], activation='sigmoid', name=layer_name)
            prev_layer = decoder_layer(prev_layer)
            prev_layer = Dropout(0.05)(prev_layer)
        # last decoder layer - corresponds to encoder_1
        last_layer_name = "decoder_1"
        # last_decoder_layer = DenseTied(input_dim, activation='sigmoid', name=last_layer_name, tied_to=encoder_layers[1])
        last_decoder_layer = DenseTranspose(dense=encoder_layers[1], activation='sigmoid', name=last_layer_name)
        output = last_decoder_layer(prev_layer)
        # construct the model
        model = Model(input, output)
        print(model.summary())
        for layer in model.layers:
            if layer.name in weights_cache.keys():
                layer.trainable = False
                # print(f"setting {layer.name} not trainnable")
                # do not load weights for decoder since the weights are tied to encoders
                if 'encoder' in layer.name:
                    layer.set_weights(weights_cache[layer.name])
        print(model.summary())
        model.compile(optimizer='adam', loss=contractive_loss)
        model.fit(input_data, input_data, batch_size=batch_size, epochs=epochs)
        # cache the trained weights
        for layer in model.layers:
            if 'encoder' in layer.name or 'decoder' in layer.name:
                weights_cache[layer.name] = layer.get_weights()
    return model

# get the encoder model from the trained model to predict on data
def get_encoder(model, input_dim, num_layers):
    input = Input(shape=(input_dim,))
    prev_layer = input
    for i in range(1, num_layers+1):
        encoder_name = f"encoder_{i}"
        prev_layer = model.get_layer(encoder_name)(prev_layer)
    return Model(input, prev_layer)
