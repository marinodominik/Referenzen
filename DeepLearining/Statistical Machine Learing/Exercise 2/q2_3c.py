import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_furthestKNNidx(x, data, K):
    distances = np.abs(data-x)
    return np.argsort(distances)[K-1] #index of furthest K nearest neighbor

def get_density(X, data, K):
    NX = X.shape[0]    
    p_X = np.zeros(NX)
    for n in range(NX):
        furthest_knn_idx = get_furthestKNNidx(X[n], data, K) #index of furthest K nearest neighbor      
        R = np.abs(data[furthest_knn_idx] - X[n])#max distance
        p_X[n] = K/NX/2/R
    return p_X

def main():
    x_train = np.sort(np.genfromtxt("nonParamTrain.txt"))
    x_test = np.sort(np.genfromtxt("nonParamTest.txt"))
    X = np.arange(-4,8,0.01)
    
    # Plotting
    plt.figure(figsize=(15,4))   
    sns.kdeplot(x_test, shade=True, label='Test data density')
    for K in [2,8,35]:
        p_X_train = get_density(X, x_train, K)
        plt.plot(X, p_X_train,label='Estimated, K='+str(K))     
    plt.ylim((0.0,0.6))
    plt.legend()
    plt.title('Density Estimation Using K-Nearest Neighbor')
    plt.ylabel('p(x)')
    plt.xlabel('x')
    plt.show()
        
if __name__ == '__main__':
    main()