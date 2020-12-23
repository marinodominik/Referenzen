import numpy as np

def get_data(name :str) -> np.asarray:
    file = open(name, 'r')
    X = []
    y=[]
    for i in file.readlines():
        line = i.split(',')
        X.append([float(x) for x in line[0:4]])
        y.append(float(line[4]))
    file.close()
    return np.array(X),np.array(y)

## 3a
X,y = get_data('iris.txt')  #read data  
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) #Normalize X
# Sanity check
print(np.mean(X, axis=0))
print(np.std(X, axis=0))

########################################################################### 3b
import matplotlib.pyplot as plt

M = X.shape[1] # No. of features
Sigma = np.cov(np.transpose(X)) # Covariance matrix
eigvals, eigvecs = np.linalg.eig(Sigma) # eigvals are already sorted

# total variance
var_total = np.sum( np.var(np.dot(X,eigvecs), axis=0) )
# cumulative variance explained
cve = np.zeros(M)
for D in np.arange(1,M+1):    
    B = eigvecs[:,0:D] # matrix of chosen eigenvectors     
    A = np.dot(X,B) # projection of the data X to it        
    var_D = np.sum( np.var(A, axis=0) ) # explained variance    
    cve[D-1] = var_D / var_total * 100 # cumul.percent variance explained
   
# Plotting
plt.plot(np.arange(1,M+1),cve)
plt.title('Number of principal components vs cumulative variance explained')
plt.xlabel('number of principal components')
plt.xticks(np.arange(1,M+1))
plt.grid()
plt.ylabel('% variance explained')
plt.show()

########################################################################### 3c
import seaborn as sns

# p=2, eigvecs calculated in 3b
B = eigvecs[:,0:2]
X_transformed = np.dot(X,B) # projection of the data X
# Plotting
plt.figure(figsize=(10,4)) 
ax = sns.scatterplot(X_transformed[:,0], X_transformed[:,1], hue=y, legend="full")
leg_handles = ax.get_legend_handles_labels()[0]
ax.legend(leg_handles, ['0: Setosa', '1: Versicolour', '2: Virginica'], title='Iris Flowers')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.show()

########################################################################### 3d
def nrmse(x, y): #normalized root mean square error between vectors X and Y
    return np.sqrt(np.mean((x-y)**2)) / np.ptp(x)

X_ori,y_ori = get_data('iris.txt')  #reread unnormalized data
means = np.mean(X_ori, axis=0)
stdevs = np.std(X_ori, axis=0)
# eigenvecs calculated in 3b
for D in np.arange(1,M+1):
    B = eigvecs[:,0:D]
    X_transformed = np.dot(X,B) # X is the normalized data
    X_backtransformed = np.dot(X_transformed, B.T) * stdevs + means #unnormalize
    print("Number of PC: " + str(D))
    for j in np.arange(M):
        print( 'x' + str(j+1) + ': ' + str(nrmse(X_ori[:,j], X_backtransformed[:,j])) )
        
########################################################################### 3e
from scipy.linalg import sqrtm
X,y = get_data('iris.txt')  #reread data, unnormalized
Sigma = np.cov(np.transpose(X))
e = 1e-5
W = np.linalg.inv(sqrtm(Sigma + e))

Y = np.dot(X,W)
print(np.round(np.cov(Y.T),3))