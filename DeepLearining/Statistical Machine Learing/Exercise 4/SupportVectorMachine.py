import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

iris = np.genfromtxt('dataSets/iris-pca.txt')
X_data, y_data = iris[:, 0:2], iris[:, 2]

def show_data(X, y):
    plt.figure()
    plt.title('Iris Dataset --> red=Setosa, yellow=Virginica')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def linear_kernel(x, y):
    return np.dot(x, y)

def plot_data_with_support_vectors(X, y, support_vectors, C):
    plt.figure()
    plt.title('Iris Dataset with Support Vectors C = {}'.format(C))
    plt.scatter(X[:, 0], X[:, 1], cmap='autumn', c=y_data)
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], label='support vectors', facecolors='none', linewidths=1,
                edgecolors='b', s=300)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def doSMV(X, y, kernel, C = 10):
    samples, features = X.shape
    K = np.zeros((samples, samples))
    for i in range(samples):
        for j in range(samples):
            K[i, j] = kernel(X[i], X[j])
    H = np.outer(y, y) * K

    # Converting into cvxopt format
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-1 * np.ones((samples)))
    G = cvxopt_matrix(np.vstack((np.diag(np.ones(samples)) * -1, np.identity(samples))))
    h = cvxopt_matrix(np.hstack((np.zeros(samples), np.ones(samples) * C)))
    A = cvxopt_matrix(y, (1, samples), 'd')
    b = cvxopt_matrix(0.0)

    #solve optimization problem for quadratic programming
    cvxopt_solvers.options['show_progress'] = False
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.ravel(sol['x'])

    # Selecting the set of indices S corresponding to non zero parameters
    S = alphas > 1e-5

    support_vectors = X[S]
    support_alphas = alphas[S]
    support_vector_y = y[S]


    w = ((y * alphas).T @ X).reshape(-1, 1)
    b = np.mean(support_vector_y - np.dot(support_vectors, w))


    #print('Alphas = ', support_alphas)
    #print('w = ', w.flatten())
    #print('b = ', b)

    plot_data_with_support_vectors(X, y, support_vectors, C)
    show_SMV_Plot(w, b, X, y, support_vectors, kernel.__name__, C)

def predict(w, X, b, margin = 0):
    return np.sign(np.dot(X, w) + b + margin)

def seperate_data(X, labels):
    return X[labels == -1], X[labels == 1]

def show_SMV_Plot(w, b, X, y, support_vectors, kernel_name, C):
    y_pred = predict(w, X, b)
    correct = np.sum(np.equal(y_pred, y.reshape(-1, 1)))
    print("{} out of {} predictions are correct".format(correct, len(y_pred)))

    X1, X2 = seperate_data(X, y)
    plt.plot(X1[:, 0], X1[:, 1], "ro", label='Setosa')
    plt.plot(X2[:, 0], X2[:, 1], "yo", label='Virginica')
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], label='support vectors', facecolors='none', linewidths=1, edgecolors='b', s=300)

    X1, X2 = np.meshgrid(np.linspace(min(X[:, 0]) - .3, max(X[:, 0]) + .3, 50), np.linspace(min(X[:, 1]) - .3, max(X[:, 1]) + .3, 50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = predict(w, X, b).reshape(X1.shape)
    plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    plt.contour(X1, X2, predict(w, X, b, 1).reshape(X1.shape), [0.0], colors='m', linewidths=1, origin='lower', )
    plt.contour(X1, X2, predict(w, X, b, -1).reshape(X1.shape), [0.0], colors='m', linewidths=1, origin='lower')
    plt.title('Support Vector Machine with ' + kernel_name + ' and C = {}'.format(C))
    plt.legend()
    plt.show()

    #missclassified points
    setosa, virginica = [], []
    for i, j, k in zip(y, y_pred, X_data):
        if(i != j):
            if i == -1:
                setosa.append(k)
            else:
                virginica.append(k)
    setosa = np.asarray(setosa)
    virginica = np.asarray(virginica)

    plt.figure()
    plt.plot(setosa[:, 0], setosa[:, 1], "ro", label='Setosa')
    plt.plot(virginica[:, 0], virginica[:, 1], "yo", label='Virginica')
    plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    plt.contour(X1, X2, predict(w, X, b, 1).reshape(X1.shape), [0.0], colors='m', linewidths=1, origin='lower', )
    plt.contour(X1, X2, predict(w, X, b, -1).reshape(X1.shape), [0.0], colors='m', linewidths=1, origin='lower')
    plt.legend()
    plt.title('Misclassified points of Setosa and Virginica')
    plt.show()


def main():
    def change_labels(labels):
        Y = []
        for y in labels:
            if y == 0:
                Y.append(-1)
            else:
                Y.append(1)
        return np.asarray(Y)

    X, y = X_data, change_labels(y_data)
    show_data(X, y)
    doSMV(X, y, linear_kernel, 10)

if __name__ == '__main__':
    main()