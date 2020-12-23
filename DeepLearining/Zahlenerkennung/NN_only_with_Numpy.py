import numpy as np
import matplotlib.pyplot as plt

# Training Dataset
Xtrain = np.genfromtxt("mnist_small_train_in.txt", delimiter=',', dtype=int)
ytrain = np.genfromtxt("mnist_small_train_out.txt", dtype=int)

# Test Dataset
Xtest = np.genfromtxt("mnist_small_test_in.txt", delimiter=',', dtype=int)
ytest = np.genfromtxt("mnist_small_test_out.txt", dtype=int)

# Check 1 training sample
plt.imshow(Xtrain[0,:].reshape((28,28)))
plt.title('Training digit:' + str(ytrain[0]))
plt.axis("off")
plt.show

## PREPROCESSING
Ntr = ytrain.shape[0] # No. of training samples
Nte = ytest.shape[0] # No. of training samples
K = 10 # No. of classes
L1 = 785 # No. of nodes in input layer
L2 = 300 # No. of nodes in hidden layer

# Add x0=1
Xtrain = np.hstack((np.ones((Ntr,1)), Xtrain))
Xtest = np.hstack((np.ones((Nte,1)), Xtest))
# One-Hot-Encoding
Ytrain = np.eye(K)[ytrain]
Ytest = np.eye(K)[ytest]

## Initialization
np.random.seed(1)
# weight matrices and bias
W1 = np.random.rand(L2,L1) / np.sqrt(L1)
W2 = np.random.rand(K,L2) / np.sqrt(L2)
bias = np.random.rand(K,1) / np.sqrt(K)

#############################################################################
def tanh_grad(a):
    return 1-(np.tanh(a))**2
def sigmoid(a):
    return np.where(a>= 0, 1/(1 + np.exp(-a)), np.exp(a)/(1 + np.exp(a)))
def sig_grad(a):
    return sigmoid(a) * (1 - sigmoid(a))
def softmax(a):
    e_a = np.exp(a - np.max(a)) # max(a) for numerical stability
    return e_a / e_a.sum(axis=0) 
def Loss(y, y_hat):
    return np.sum(y * np.log(y_hat + 1e-6))  # add 1e-6 for numerical stability

## TRAINING & TESTING
## =>> WARNING!! SLOW!! NOT OPTIMIZED WHATSOVER, Works though!
epochs = 20000 # How many complete passes through the training set

batchsize = 256 # just because
alpha1 = 1 # Learning rate
alpha2 = 0.1 # Learning rate
dL_dW1 = 0
dL_dW2 = 0
i=0
train_loss = []
TR = []
TE = []
while i<epochs:
    idx = np.random.permutation(Ntr)
    X = Xtrain[idx,:]
    Y = Ytrain[idx,:]
    
    # Train with mini batches
    for start_idx in range(0, Ntr, batchsize):
        end_idx = min(start_idx+batchsize, Ntr)
        
        # mini batch
        Xb = np.transpose(X[start_idx:end_idx,:])
        Yb = np.transpose(Y[start_idx:end_idx,:])
        
        minibatch_grads1 = np.empty(W1.shape)
        minibatch_grads2 = np.empty(W2.shape)
        minibatch_gradsb = np.empty((K,1))
        minibatch_loss = np.empty(Xb.shape[1])
        for b in np.arange(Xb.shape[1]):            
            # Forward feed
            # input => hidden layer
            Z1 = np.dot(W1, Xb[:,b])[:, np.newaxis]
            #a1 = sigmoid(Z1)
            a1 = np.tanh(Z1)
            # hidden layer => output
            Z2 = bias + np.matmul(W2,a1)
            Y_hat = softmax(Z2)
            
            # Backward feed/ Gradients
            dL_dz2 = (Y_hat - Yb[:,b][:, np.newaxis])
            #da1_dz1 = sig_grad(Z1)
            da1_dz1 = tanh_grad(Z1)
            dL_dW2 = np.matmul( dL_dz2 , a1.T )
            dL_dW1 = np.matmul( (np.matmul(W2.T, dL_dz2) * da1_dz1), np.transpose(Xb[:,b][:, np.newaxis]))
            
            minibatch_grads1 += dL_dW1
            minibatch_grads2 += dL_dW2
            minibatch_gradsb += dL_dz2
            #minibatch_loss[b] = Loss(Yb[:,b], Y_hat)
        
    # Update
    dW1 = minibatch_grads1 / Xb.shape[1]
    dW2 = minibatch_grads2 / Xb.shape[1]
    db = minibatch_gradsb / Xb.shape[1]
    #train_loss.append(np.mean(minibatch_loss))
    W1 = W1 - alpha1 * dW1
    W2 = W2 - alpha2 * dW2
    bias = bias - alpha2 * db
        
    # Training error
    tr_e = 0
    for n in range(Ntr):
        x = Xtrain[n,:].T
        Z1 = np.dot(W1, x)[:, np.newaxis]
        #a1 = sigmoid(Z1)
        a1 = np.tanh(Z1)
        Z2 = bias + np.matmul(W2,a1)
        Y_hat = softmax(Z2)
        if np.argmax(Y_hat)!=ytrain[n]:
            tr_e += 1
    
    # Test error
    te_e = 0
    for n in range(Nte):
        x = Xtest[n,:].T
        Z1 = np.dot(W1, x)[:, np.newaxis]
        #a1 = sigmoid(Z1)
        a1 = np.tanh(Z1)
        Z2 = bias + np.matmul(W2,a1)
        Y_hat = softmax(Z2)
        if np.argmax(Y_hat)!=ytest[n]:
            te_e += 1
     
    print("epoch #"+str(i+1)+" Tr err: "+str(tr_e) +", Te err: "+str(te_e))          
    TR.append(tr_e)
    TE.append(te_e)
    i+=1

TR_perc = np.array(TR)/Ntr*100
TE_perc = np.array(TE)/Nte*100
plt.figure(figsize=(15,4)) 
plt.plot(np.arange(1,TR_perc.shape[0]+1), TR_perc, label="Training error percentage")
plt.plot(np.arange(1,TE_perc.shape[0]+1), TE_perc, label="Test error percentage")
plt.yticks(np.arange(0,90,10))
plt.xlabel("Epochs #")
plt.ylabel("%")
plt.legend()
plt.grid('on')
plt.show()

plt.figure(figsize=(15,4)) 
plt.plot(np.arange(601,TR_perc.shape[0]+1), TR_perc[600:], label="Training error percentage")
plt.plot(np.arange(601,TE_perc.shape[0]+1), TE_perc[600:], label="Test error percentage")
plt.yticks(np.arange(0,10,1))
plt.xlabel("Epochs #")
plt.ylabel("%")
plt.legend()
plt.title("Closer look")
plt.grid('on')
plt.show()
