import sys
import numpy as np
import matplotlib.pyplot as plt


def mutation(n = 40):
    P = np.array([[0.45, 0.55], [0.023, 0.977]])
    x = np.array([1, 0]).T
    x1 = []
    x2 = []
    y = []
    x_last = 0
    for i in range(n):
        p = x @ np.linalg.matrix_power(P, i)
        x1.append(p[0])
        x2.append(p[1])
        y.append(i)

        if np.array_equal(x_last, p):
            print(i)
            break
        x_last = p



    plt.plot(y, x1, label='mutation')
    plt.xlabel("iteration")
    plt.ylabel('Probability')

    plt.plot(y, x2, label='no mutation')
    plt.xlabel("iteration")
    plt.ylabel('Probability')
    plt.legend()
    plt.show()


def rosen_function(X: np.ndarray) -> float:
    """
    #sum(i to i-1) [100 * (x_i+1 - x_i)^2 + (x_i - 1)^2]
    :param X: 1D array of points at which the Rosenbrock function is to be computed.
    :return: float: The value of the Rosenbrock function
    """
    rosen = 0
    for i in range (X.shape[0] - 1):
        rosen += 100 * (X[i+1] - X[i]**2.0) ** 2.0 + (X[i] - 1) ** 2.0
    return rosen


def rosen_derivative(X) -> np.ndarray:
    """
    :param X: 1D array of points at which the derivative is to be computed
    :return: (N,) ndarray The gradient of the Rosenbrock function at `x`.
    """
    df = [400 * X[0] * (X[0] ** 2 - X[1]) + 2 * (X[0] - 1)]
    for i in range(1, X.shape[0] - 1):
        dx = 400 * X[i] * (X[i]**2 - X[i+1]) + 2 * (X[i] - 1)
        dy = 200 * (X[i] - X[i-1] ** 2)
        df.append(dx + dy)

    df.append(200 * (X[X.shape[0] - 1] - X[X.shape[0] - 2] ** 2))
    return np.asarray(df)

def main():
    #inizialization
    N = 10000
    n = 20
    learning_rate = 0.001
    X = np.linspace(0, 1, n)

    rosen = []
    y = []
    for i in range(N):
        r = rosen_function(X)
        rosen.append(r)
        df_rosen = rosen_derivative(X)
        X = X - learning_rate * df_rosen
        y.append(i)
    plt.plot(y, rosen, label='learning rate = 0.001')

    learning_rate = 0.002
    X = np.linspace(0, 1, n)
    rosen = []
    y = []
    for i in range(N):
        r = rosen_function(X)
        rosen.append(r)
        df_rosen = rosen_derivative(X)
        X = X - learning_rate * df_rosen
        y.append(i)

    plt.plot(y, rosen, label="learning rate = 0.002")

    learning_rate = 0.0001
    X = np.linspace(0, 1, n)
    rosen = []
    y = []
    for i in range(N):
        r = rosen_function(X)
        rosen.append(r)
        df_rosen = rosen_derivative(X)
        X = X - learning_rate * df_rosen
        y.append(i)

    plt.plot(y, rosen, label="learning rate = 0.0001")


    plt.title("Learning Curve of Rosenbrock function")
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #main()
    mutation()
    sys.exit()