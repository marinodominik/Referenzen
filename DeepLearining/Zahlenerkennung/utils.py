import numpy as np
import matplotlib.pyplot as plt

def changeNum2Vec(data):
    y = []
    for value in data:
        vec = np.zeros(10)      #zero vector
        vec[int(value)] = 1.0
        y.append(vec)
    return np.asarray(y)

def changeVec2Num(data):
    y = []
    for vec in data:
        if np.equal(vec, np.zeros(10)).all():
            y.append(-1)
        else:
            for index, v in enumerate(vec):
                if v == 1:
                    y.append(index)

    return np.asarray(y)

def getData(path_X, path_y):
    X = np.loadtxt(path_X, delimiter=',')
    y = np.loadtxt(path_y, delimiter=',')
    return np.c_[X, np.ones(len(y))], y        #added bias to X in last column


def get_random_batch(X, y, size):
    perm = np.random.permutation(len(y))
    X = X[perm]             #randomize Trainset
    y = y[perm]             #randomize Trainset

    X_batches = np.array_split(X, len(y)//size)
    y_batches = np.array_split(y, len(y)//size)

    return X_batches, y_batches

def visualization_error(epoch, data_train, data_test, title):
    plt.figure()
    plt.plot(epoch, data_train, color='b', label='train')
    plt.plot(epoch, data_test, color='r', label="test")
    plt.title(title)
    plt.xlabel("epochs")
    plt.ylabel('%')
    plt.legend()
    plt.grid('on')
    plt.show()


def zoom_smallest_error(error):
    smallest_error = 0
    index_of_smallest_error = 0
    for index, e in enumerate(error[:, 1]):
        if index == 0:
            smallest_error = e
            index_of_smallest_error = index
        elif e < smallest_error:
            smallest_error = e
            index_of_smallest_error = index

    min = index_of_smallest_error - 20
    max = index_of_smallest_error + 20
    if min < 0:
        min = 0
    if max > len(error):
        max = len(error)
    new_error_zoom = error[min:max, :]

    epoch = []
    minimum = min * 5
    for i in range(min, max, 1):
        epoch.append(minimum)
        minimum += 5

    return np.asarray(new_error_zoom), smallest_error, index_of_smallest_error * 5, epoch

def save_nparray(data, file_name):
    np.save(file_name, data)

def load_nparray(file_name):
    data = np.load(file_name)
    return data

#TODO save numpy array in .txt file i.e train and test error --> lernen mit 1000 epochen dauert ca. 13 minuten