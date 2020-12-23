from tqdm import tqdm
import numpy as np
import utils


###############################################################################
########################## DATA ###############################################
X_train, y_train = utils.getData('Data/mnist_small_train_in.txt', 'Data/mnist_small_train_out.txt')
X_test, y_test = utils.getData('Data/mnist_small_test_in.txt', 'Data/mnist_small_test_out.txt')

y_train = utils.changeNum2Vec(y_train)
y_test = utils.changeNum2Vec(y_test)
######################### END DATA #############################################
################################################################################



def train(sees, model, params):
    loss, accuracy, error = [], [], []
    eval_every_ith_epoch = 5
    for epoch in tqdm(range(params['epochs'])):
        X_train_batch, y_train_batch = utils.get_random_batch(X_train, y_train, params['batch_size'])

        for X_batch, y_batch in zip(X_train_batch, y_train_batch):
            _ = sees.run(model.optimizer, feed_dict={model.X: X_batch, model.t: y_batch})

        #evaluation every 5 iteration
        if epoch % eval_every_ith_epoch == 0:
            loss_train, acc_train, error_train = evaluation(sees, model, X_train, y_train)
            loss_test, acc_test, error_test = evaluation(sees, model, X_test, y_test)

            loss.append([loss_train, loss_test])
            accuracy.append([acc_train, acc_test])
            error.append([error_train, error_test])

    return np.asarray(loss), np.asarray(accuracy), np.asarray(error)


def evaluation(sess, model, X, y):
    loss, y_pred = sess.run([model.loss, model.y], feed_dict={model.X: X, model.t: y})
    accuracy = np.sum(np.equal(np.rint(y_pred), y)) / (len(y))

    error = 0
    for y_vec, y_pred_vec in zip(y, np.rint(y_pred)):
        if not (y_vec == y_pred_vec).all():
            error += 1

    return loss, accuracy, (error / len(y) * 100)

