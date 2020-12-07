import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from time import sleep
import sys
from sklearn.cluster import KMeans
from scipy.stats import chi2

def mahalanobis(W1, W2, K):
    """
    W1 = data from window 1
    W2 = data from window 2
    K = number of clusters.
    """
    
    # m1 - number of rows sampled
    # n - number of features (should be same for W1 and W2)
    m1, n = W1.shape
    m2 = W2.shape[0]
    
    # Using K-means for clustering since Gaussian will be computationally very expensive
    kmeans = KMeans(n_clusters = K).fit(W1)
    
    # means of each cluster to calculate the probability function of each cluster
    labels = kmeans.predict(W1)
    means = kmeans.cluster_centers_
    
    # Common covariance matrix initialisation for each cluster
    cov_store = np.zeros((n*n,K))
    weights = np.zeros((K))
    
    # weight calculation for each cluster
    for i in range(K):
        w = W1[labels == i].shape[0]
        weights[i] = w
        if (w > 1):
            co = np.cov(W1[labels == i].T)
            cov_store[:,i] = co.flatten()
    
    weights /= m1
    
    sc = np.sum(cov_store*(np.tile(weights, (n*n,1))),1)
    
    # Inverse of covariance calculation for mahalanobis distance
    sc = np.reshape(sc, (n,n))
    invcov = np.linalg.pinv(sc)
    
    covsum = np.zeros((m2))
    
    # Calculate the Mahalanobis distance
    for i in range(m2):        
        dist = np.zeros((K))
        for k in range(K):
            if weights[k] > 0:
                dist[k] = np.dot(np.dot(means[k,:] - W2[i,:] , invcov), means[k,:] - W2[i,:])
            else:
                dist[k] = np.inf
        covsum[i] = np.min(dist)
    
    # Return the mean of variance of each cluster
    return np.mean(covsum)


def SPLL(W1,W2,K):
    
    n = W1.shape[1]
    
    s1 = mahalanobis(W1, W2, K)
##    s1 = 0
    s2 = mahalanobis(W2, W1, K)
    
    s = max(s1,s2)
    # Chi square value for n degree of freedoms
    p_val = chi2.cdf(s,n)
    # Two-tailed chi square test
    ps = min (p_val, 1 - p_val)
    
    # Considering the confidence interval to be 0.05
    return int(ps < alpha)#, ps, s

def data_generator(x1,x2,x3,start_mean,start_var,i):
#     mean_change = input()
#     var_change = input()
    global trend;
    change = 0
    if (i%300 < 30) and (i%300 > 10):
        mean_change = start_mean+4
        var_change = start_var+2
        change = 1
    else:
        mean_change = start_mean
        var_change = start_var
    return (np.append(x1,np.random.normal(mean_change + trend*i,var_change)),
            np.append(x2,np.random.normal(mean_change + 3 - trend*i, var_change)),
            np.append(x3,np.random.normal(mean_change - 4 + trend*i, var_change+1)))


window_size = 40
start_mean = 4#int(input('Enter mean:'))
start_var = 1#int(input('Enter variance:'))
display = 100
alpha = 0.05
k = 5
offline_num = 100
trend = 0.001
n = 3000

# Time intervals
x = np.array([i for i in range(window_size)])

# Features
x1 = np.random.normal(start_mean, start_var, window_size)
x2 = np.random.normal(start_mean + 3,start_var, window_size)
x3 = np.random.normal(start_mean - 4, start_var + 1, window_size)


# Sliding Windows
base_window = np.zeros([40,3])
base_window[:,0] = x1[-window_size:]
base_window[:,1] = x2[-window_size:]
base_window[:,2] = x3[-window_size:]
current_window = np.zeros([40,3])
change = 0
change_stack = np.array([])
f_change = 0
val = 0
icrl = []
false_alarms = 0

fig = plt.figure(figsize = (12,9))
ax1 = fig.add_subplot(4,1,1)

ax2 = fig.add_subplot(4,1,2)

ax3 = fig.add_subplot(4,1,3)

ax4 = fig.add_subplot(4,1,4)

def animate(i):

    global window_size, start_mean, start_var, display,alpha, k,change_stack, icrl, false_alarms, \
    x,x1,x2,x3,base_window, current_window,fig, ax1, ax2, ax3, ax4, change,f_change, val;
    if i != 0:    
        print ("output ",i)
    ##    print ("before")
    ##    print (base_window)
        # Change detection algo
##        print (change_stack.size, np.sum(change_stack[-window_size*2:-window_size]))
        if (change_stack.size >= 2*window_size) and (np.sum(change_stack[-window_size*2:]) == 0):
##                print ('entered')
    ##            print (x1[-window_size*2:-window_size-1].size )
    ##            print (x1)
                base_window[:,0] = x1[-window_size*2:-window_size]
                base_window[:,1] = x2[-window_size*2:-window_size]
                base_window[:,2] = x3[-window_size*2:-window_size]
    ##    
        current_window[:,0] = x1[-window_size:]
        current_window[:,1] = x2[-window_size:]
        current_window[:,2] = x3[-window_size:]
    
    ##    print ("mid1")
    ##    print (current_window,base_window)

        # Run the SPLL algo for given alpha and k
        change = SPLL(base_window, current_window, k)

    ##    print ("mid2")
    ##    print (current_window,base_window)
##        print (change)
        # Plotting
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax1.plot(x[-display:],x1[-display:],'b')
        ax1.plot(x[-display:],x2[-display:],'y')
        ax1.plot(x[-display:],x3[-display:],'g')
        ax2.plot(x[-display:],x1[-display:],'b')
        ax3.plot(x[-display:],x2[-display:],'y')
        ax4.plot(x[-display:],x3[-display:],'g')
        if change:
            ax1.plot(x[-1],x1[-1],'ro')
            ax1.plot(x[-1],x2[-1],'ro')
            ax1.plot(x[-1],x3[-1],'ro')
            ax2.plot(x[-1],x1[-1],'ro')
            ax3.plot(x[-1],x2[-1],'ro')
            ax4.plot(x[-1],x3[-1],'ro')
        else:
            ax1.plot(x[-1],x1[-1],'bo')
            ax1.plot(x[-1],x2[-1],'yo')
            ax1.plot(x[-1],x3[-1],'go')
            ax2.plot(x[-1],x1[-1],'bo')
            ax3.plot(x[-1],x2[-1],'yo')
            ax4.plot(x[-1],x3[-1],'go')
        ax1.set_ylabel("Combined")
        ax1.set_xlabel("Time")
        ax2.set_ylabel("x1")
        ax2.set_xlabel("Time")
        ax3.set_ylabel("x2")
        ax3.set_xlabel("Time")
        ax4.set_ylabel("x3")
        ax4.set_xlabel("Time")
        
        x = np.append(x,i+window_size)[-display:]
        
        x1,x2,x3 = data_generator(x1,x2,x3,start_mean,start_var,i)


        if (i%300 > 10) and (i%300 < 30 + window_size):
            f_change += 1 - change
            val += 1
        elif (i%300 == 30 + window_size):
            icrl.append(f_change)
            print (f_change, val)
            f_change = 0
            val = 0

        if (val == 0) and (change == 1):
            false_alarms += 1
            print ('False alarm', false_alarms)

##        print (change)
        # Stack to store the change values
        change_stack = np.append(change_stack,change)
        if change_stack.size > offline_num:
            change_stack = np.delete(change_stack,0)
    
##    print ("after")
##    print (base_window)
    
    
ani = animation.FuncAnimation(fig, animate,frames = n + 1, repeat = False,  interval = 1)
plt.show()
fun = lambda x: x/59
print ("ARL mean", np.mean(icrl))
print ("ARL rate", np.mean(list(map(fun,icrl))))
print ("False Alarm numbers =", false_alarms)
