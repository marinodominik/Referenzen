import numpy as np
import matplotlib.pyplot as plt

def kernel(xi, xj, sigma_f=1.0, l=1):
    return sigma_f * np.exp(-(np.linalg.norm(xi - xj)**2) / (2 * l**2))

def f(x):
    return np.sin(x) + np.sin(x)**2

def Cnew_comps(x, xnew, sigma_f=1.0, l=1): #Calculate the components of of Cnew
    n = x.shape[0]
    nnew = xnew.shape[0]
    
    Cn = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            Cn[i,j] = kernel(x[i], x[j], sigma_f=sigma_f, l=l)
            
    k = np.empty((nnew, n))
    for i in range(nnew):
        for j in range(n):
            k[i,j] = kernel(xnew[i], x[j], sigma_f=sigma_f, l=l)
            
    c = np.empty((nnew, nnew))
    for i in range(nnew):
        for j in range(nnew):
            c[i,j] = kernel(xnew[i], xnew[j], sigma_f=sigma_f, l=l)
            
    return (Cn, k, c)

def gpr_params(Cn, c, k, sigma_n, t): # gaussian regression parameters
    n = Cn.shape[0]
    # mean
    m = np.dot(k, np.dot(np.linalg.inv(Cn + (sigma_n**2)*np.eye(n)), t.reshape([n, 1])))
    # Covariance.
    S = c - np.dot(k, np.dot(np.linalg.inv(Cn + (sigma_n**2)*np.eye(n)), k.T))    
    return (m, S)

np.random.seed(1)
# prediction points 
xnew = np.arange(0,2,0.005) * np.pi
nnew = xnew.shape[0]
# error standard deviation
sigma_n = np.sqrt(0.001)
# draw error randomly 
epsilon = np.random.normal(0, sigma_n, nnew)
# observed
t_observed = f(xnew) + epsilon

# start with no target data point
x = np.array([])
t = np.array([])
# start with mean=0
mnew = np.zeros((1,1))
uncertainty = np.random.rand(nnew)
for i in range(16):
    newptidx = np.argmax(uncertainty)
    x = np.append(x,xnew[newptidx])
    t = np.append(t,t_observed[newptidx])
    n = x.shape[0]
    
    # Cov matrix
    l = 1
    sigma_f = 1
    Cn, k, c = Cnew_comps(x, xnew, sigma_f=sigma_f, l=l)
    Cnew_left = np.concatenate((Cn + (sigma_n**2)*np.eye(n), k), axis=0)
    Cnew_right = np.concatenate((k.T, c), axis=0)
    Cnew = np.concatenate((Cnew_left, Cnew_right), axis=1)
    # GPR parameters
    mnew, covnew = gpr_params(Cn, c, k, sigma_n, t)
    # Sample from multivariate Gaussian with the GPR parameters
    t_pred = np.random.multivariate_normal(mean=mnew.ravel(), cov=covnew)
    # Define uncertainty=2stdev 
    uncertainty = 2 * np.sqrt(np.diag(covnew))
    
    if i in [0,1,3,7,15]: 
        plt.figure(figsize=(8,4)) 
        plt.plot(xnew, t_pred, label='Prediction')    
        plt.fill_between(xnew, mnew.ravel() + uncertainty, mnew.ravel() - uncertainty, alpha=0.1)
        plt.plot(xnew, f(xnew), '--', color='black', label="Ground Truth", alpha=0.8)
        plt.scatter(xnew, t_observed, color='orange', marker='.', linewidths=0.5, alpha=0.5, label="f(x) + noise")
        plt.scatter(x, t, color='red', marker='x', linewidths=5, label='Samples')
        plt.xlabel('x')
        plt.title("Iteration #" + str(i+1))
        plt.legend()
        plt.show()
