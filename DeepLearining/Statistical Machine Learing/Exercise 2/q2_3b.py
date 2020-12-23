import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def H_gauss(u): #1-D Gaussian kernel
    return np.exp(-(u)**2/2) / np.sqrt(2*np.pi)

def K_gauss(X, data, h):
    M = X.shape[0]
    N = data.shape[0]
    K = np.zeros(M)
    for m in range(M):
        K_sum = 0
        for n in range(N):
            K_sum += H_gauss((X[m]-data[n])/h)
        K[m] = K_sum
    return K

def LL(p):
    return np.sum(np.log(p))
    
def main():
    x_train = np.genfromtxt("nonParamTrain.txt")
    x_test = np.genfromtxt("nonParamTest.txt")
    Ntrain = x_train.shape[0]
    Ntest = x_test.shape[0]
    X = np.arange(-4,8,0.01)
    
    ll_train = np.zeros(3)
    ll_test = np.zeros(3)
    i = 0
    
    plt.figure(figsize=(15,4))
    sns.kdeplot(x_test, shade=True, label='Test data density')
    for h in [0.03, 0.2, 0.8]:    
        p_train = K_gauss(x_train, x_train, h) / Ntrain / h
        p_test = K_gauss(x_test, x_test, h) / Ntest / h
        p_X_train = K_gauss(X, x_train, h) / Ntrain / h

        # Log likelihood
        ll_train[i] = LL(p_train)
        print('Log likelihood of the training data with h='+str(h)+' is '+str(ll_train[i]))
        ll_test[i] = LL(p_test)
        print('Log likelihood of the test data with h='+str(h)+' is '+str(ll_test[i]))     
        i += 1
        
        # Plotting  
        plt.plot(X, p_X_train, label='Estimated, h='+str(h))
    plt.ylim((0.0,0.7))
    plt.legend()
    plt.title('Density Estimation Using Gaussian Kernel')
    plt.ylabel('p(x)')
    plt.xlabel('x')
    plt.show()
        
if __name__ == '__main__':
    main()
