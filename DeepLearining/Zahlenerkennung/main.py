import Model
import TrainTestMLP as mlp
import tensorflow as tf
import utils
import numpy as np


loss = []
def train_test_mlp(model, params):
    tf.compat.v1.set_random_seed(42)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        loss, accuracy, error = mlp.train(sess, model, params)
    sess.close()
    tf.compat.v1.reset_default_graph()

    return loss, accuracy, error

def main():
    parameter = {
        "dimension": 28*28,
        "classes": 10,
        "epochs": 1000,
        "learning_rate": 0.01,
        "hidden_layers": [(300, tf.nn.relu, False)],
        #'hidden_layers': [(512, tf.nn.relu, False), (256, tf.nn.relu, False), (128, tf.nn.relu, False), (32, tf.nn.relu, False)],
        "output_activation": tf.nn.softmax,
        "batch_size": 10
    }

    model = Model.Model(parameter)
    loss, accuracy, error = train_test_mlp(model, parameter)

    utils.visualization_error([i for i in range(0, parameter['epochs'], 5)], accuracy[:, 0], accuracy[:, 1], "Train/Test Accuracy in Percentage", )
    utils.visualization_error([i for i in range(0, parameter['epochs'], 5)], loss[:, 0], loss[:, 1], "Train/Test loss in Percentage")
    utils.visualization_error([i for i in range(0, parameter['epochs'], 5)], error[:, 0], error[:, 1], "Train/Test error in Percentage")

    utils.save_nparray(error, "error.npy")
    error, small_error, index_of_error, epoch = utils.zoom_smallest_error(error)
    print("The index of the smallest error {} is {}".format(small_error, index_of_error))
    utils.visualization_error(epoch, error[:, 0], error[:, 1], "Train/Test error in Percentage")


if __name__ == '__main__':
    main()