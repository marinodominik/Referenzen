import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def delta(x,InvS,mu, pi):
    d = np.matmul(np.matmul(np.transpose(x), InvS),mu) -0.5 * np.matmul(np.matmul(np.transpose(mu), InvS), mu) + np.log(pi)
    return d

x = np.genfromtxt("ldaData.txt")

# No. of samples per class
N1 = 50
N2 = 43
N3 = 44
N = x.shape[0]

# data per class
x1 = x[0:N1,:]
x2 = x[N1:N1+N2,:]
x3 = x[N1+N2:N1+N2+N3,:]

# class vector
c = np.concatenate((np.repeat(1,50), np.repeat(2,43), np.repeat(3,44)))

plt.figure(figsize=(15,4)) 
sns.scatterplot(x[:,0], x[:,1], hue=c, style=c, palette='Set1', legend='full')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('Given Data Set & Classification')
plt.legend()
plt.show()

# priors
pi1 = N1/N
pi2 = N2/N
pi3 = N3/N

# The class means
mu1 = np.mean(x1, axis=0).reshape((2,1))
mu2 = np.mean(x2, axis=0).reshape((2,1))
mu3 = np.mean(x3, axis=0).reshape((2,1))

# Common Sigma
S = np.cov(np.transpose(x))
InvS = np.linalg.inv(S)

# Classifier
cnew = np.zeros(N, dtype='int8')
for n in range(N):
    d1 = delta(x[n,:], InvS, mu1, pi1)
    d2 = delta(x[n,:], InvS, mu2, pi2)
    d3 = delta(x[n,:], InvS, mu3, pi3)
    cnew[n] = np.argmax((d1, d2, d3)) + 1
    
# Plot new classification result
plt.figure(figsize=(15,4)) 
sns.scatterplot(x[:,0], x[:,1], hue=cnew, style=cnew, palette='Set1', legend='full')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('Estimated Classification using LDA')
plt.legend()
plt.show()

print('Misclassified points: ' + str(np.sum(c!=cnew)))

wrongidx = np.argwhere(c!=cnew)
for i in range(len(wrongidx)):
    print('pt (' + str(x[wrongidx[i],0]) + ','+ str(x[wrongidx[i],1]) + 
                  '), given: ' + str(c[wrongidx[i]]) +
            ', classified as: ' + str(cnew[wrongidx[i]]))
