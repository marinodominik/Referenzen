import numpy as np
import matplotlib.pyplot as plt
    
def get_counts(x, binsize):    
    xmin = min(x)
    xmax = max(x)
    which_bin = np.floor((x-xmin)/binsize)
    edges = np.arange(xmin, xmax + binsize, binsize)
    Nbins = edges.shape[0]-1
    counts = np.zeros(Nbins)
    for n in range(Nbins):
        counts[n] = np.count_nonzero(which_bin==n)
    return (edges[1:]-binsize/2), counts
    
def plot_my_hist(x, y, binsize, title): # To plot two distributions side by side
    fig, axs = plt.subplots(1,2)
    fig.suptitle(title) 
    
    midpts, counts = get_counts(x, binsize)
    axs[0].bar(x=midpts, height=counts, width=binsize)    
    
    midpts, counts = get_counts(y, binsize) 
    axs[1].bar(x=midpts, height=counts, width=binsize)
    
    plt.show()
    
def main():
    x_train = np.genfromtxt("nonParamTrain.txt")
    x_test = np.genfromtxt("nonParamTest.txt")
    
    for b in [0.02, 0.5, 2.0]:
        plot_my_hist(x_train, x_test, b, "Histogram of x_ train and x_test with bin size = "+ str(b))
    
if __name__ == '__main__':
    main()