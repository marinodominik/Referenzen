import tensorflow as tf

bias = 1

class Model:
    def __init__(self, parameters):
        tf.compat.v1.disable_v2_behavior()

        self.X = tf.compat.v1.placeholder(dtype=tf.float16, shape=[None, parameters['dimension'] + bias], name='InputData') #shape = [1x784] = [1, 28*28] + bias
        self.t = tf.compat.v1.placeholder(dtype=tf.float16, shape=[None, parameters['classes']], name='LabelData')          #digit form 0 to 9

        layer_chain = [self.X]      #inputlayer
        for params in parameters['hidden_layers']:
            number_of_units, activation, regularizer = params
            layer = tf.compat.v1.layers.dense(layer_chain[-1],
                                              number_of_units,
                                              activation,
                                              use_bias = True,
                                              #kernel_regularizer=regularizer,
                                              kernel_initializer=tf.random_normal_initializer())
            layer_chain.append(layer)

        self.y = tf.compat.v1.layers.dense(layer_chain[-1], parameters['classes'], parameters['output_activation'])         #output layer

        self.loss = tf.compat.v1.losses.mean_squared_error(self.t, self.y)      #TODO falsche loss function
        self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(parameters['learning_rate']).minimize(self.loss)

        #TODO saving model and load model after learning