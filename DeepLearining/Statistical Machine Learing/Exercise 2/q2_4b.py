import numpy as np
import matplotlib.pyplot as plt

def get_data(name :str) -> np.asarray:
    file = open(name, 'r')
    x1 = []
    x2 = []
    for i in file.readlines():
        line = i.split()
        x1.append(float(line[0]))
        x2.append(float(line[1]))
    file.close()
    return np.row_stack((x1,x2))

def get_cov(X):
    return np.matmul(X,np.transpose(X))/X.shape[1]

def multvar_gauss(x,m,S):
    p = m.shape[0]
    return (2*np.pi)**(-p/2) / np.sqrt(np.linalg.det(S)) * np.exp(-0.5*np.matmul(np.transpose(x-m), np.matmul(np.linalg.inv(S),(x-m)) ))

def get_mixpdf(x1,x2,t):
    X=np.array([x1,x2])
    M = int(len(t) / 3)
    pdf = 0
    for j in range(M):
        m = t[j*3]
        S = t[j*3+1]
        pi = t[j*3+2]
        pdf += pi * multvar_gauss(X,m,S)
    return pdf

def plot_overlay(x,t,title): 
    nbins = 50
    data_density = plt.hist2d(x[0,:],x[1,:],bins=nbins)
    x1_edges = data_density[1]
    x2_edges = data_density[2]
    X1,X2 = np.meshgrid(x1_edges, x2_edges)
    mixpdf = np.zeros((nbins+1,nbins+1))
    for i in range(nbins+1):
        for j in range(nbins+1):
            mixpdf[i,j] = get_mixpdf(X1[i,j],X2[i,j],t)
    plt.contour(X1, X2, mixpdf, cmap="Wistia")
    plt.title(title)
    plt.show()
    
def main():
    #read data
    x = get_data('gmm.txt')
    p = x.shape[0]
    N = x.shape[1]
    
    #initialization
    M = 4
    pi = 1/M
    all_mu = [None]*M
    all_Sigma = [None]*M
    all_pi = [pi]*M
    np.random.seed=1234
    for j in range(M):
        all_mu[j] = np.random.rand(2)
        all_Sigma[j] = get_cov(x-all_mu[j][:,np.newaxis])
    
    alpha = np.zeros((M,N))
    steps = 30
    theta = [None]*(steps) #tracking all the values for each step {mu_1, Sigma_1, pi_1, ...}
    L = np.zeros(steps) #log likelihood
    
    for i in range(steps): 
        t = [None] * (3*M)
        
        #E-Step
        for n in range(N):
            alpha_denom = 0
            beta = np.empty(M)
            for j in range(M):
                bn = multvar_gauss(x[:,n],all_mu[j],all_Sigma[j])
                beta[j] = all_pi[j]*bn
                alpha_denom += beta[j]
                
            alpha[:,n] = beta/alpha_denom
            
        #M-step
        Nj = np.sum(alpha, axis=1)
        for j in range(M):
            all_mu[j] = np.sum(alpha[j,:]*x, axis=1) / Nj[j]
            s = np.zeros((p,p))
            for n in range(N):
                X = x[:,n]-all_mu[j]
                X = X[:,np.newaxis]
                s = s + alpha[j,n]*np.matmul(X,np.transpose(X))
            all_Sigma[j] = s / Nj[j]
        all_pi = Nj / N
      
        t[0:3*M:3] = all_mu 
        t[1:3*M:3] = all_Sigma 
        t[2:3*M:3] = all_pi
        theta[i] = t
        
        #log likelihood
        ll = 0
        for n in range(N):
            l = 0
            for j in range(M):
                l += all_pi[j] * multvar_gauss(x[:,n],all_mu[j],all_Sigma[j])
            ll += np.log(l)
        L[i] = ll
    
    for i in [1,3,5,10,30]:
        plot_overlay(x,theta[i-1], 'Data density vs mixture model for ti='+ str(i))   
    
    # plot log likelihood
    plt.plot(L)    
    plt.ylabel("Log likelihood")
    plt.xlabel("# iterations")
    plt.title("Log likelihood updates")
    plt.show()
    
    # plot mixture scales
    for j in range(M):
        pi = [x[j*3+2] for x in theta]
        plt.plot(np.arange(1,31), pi, label='pi'+ str(j+1))
    plt.ylabel("pi")
    plt.xlabel("# iterations")
    plt.title("Updates of mixture scaling parameter")
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()