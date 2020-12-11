import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation,gridspec
from time import sleep
import sys
from sklearn.cluster import KMeans
from scipy.stats import chi2
from scipy.stats import t as t_value

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

# Critical value 

def calculate_critical_value(size, alpha):
    ''' Arguments:
            size - size of the numpy array.
            aplha - the confidence interval choosen.'''
    
    t_dist = t_value.ppf(1 - alpha / (2 * size), size - 2)
    numerator = (size - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
    critical_value = numerator / denominator
##    print("Grubbs Critical Value: {}".format(critical_value))
    return critical_value

# Calculated Grubbs value

def grubbs_stat(y): 
    ''' Arguments:
            y - numpy array of which outlier is to be detected.'''
    
    std_dev = np.std(y)
    avg_y = np.mean(y)
    abs_val_minus_avg = abs(y - avg_y)
    max_of_deviations = max(abs_val_minus_avg)
    max_ind = np.argmax(abs_val_minus_avg)
    Gcal = max_of_deviations/ std_dev
##    print("Grubbs Statistics Value : {}".format(Gcal))
    return Gcal, max_ind

# Checking if outlier present
def check_G_values(Gs, Gc, inp, max_index):
    ''' Arguments:
            Gs - Grubbs statistic value or the calculated value.
            Gc - Grubbs critical value.
            inp - input numpy array.
            max_index - the index of the element with maximum deviation.'''
    
    if Gs > Gc:
##        print('{} is an outlier. G > G-critical: {:.4f} > {:.4f} \n'.format(inp[max_index], Gs, Gc))
        return True
    else:
##        print('{} is not an outlier. G > G-critical: {:.4f} > {:.4f} \n'.format(inp[max_index], Gs, Gc))
        return False

def ESD_Test(input_series, alpha, max_outliers = np.inf):
    ''' Arguments:
            input_series - input pandas dataframe.
            alpha - desired confidence interval.
            max_outliers - maximum number of outliers to delete {default: Till no outlier}'''
    outliers_list = pd.DataFrame(columns = input_series.columns)
    for j in input_series.columns:
        i = 0
        while (i < max_outliers):
            Gcritical = calculate_critical_value(input_series.shape[0], alpha)
            Gstat, max_index = grubbs_stat(input_series[j].to_numpy())
            if check_G_values(Gstat, Gcritical, input_series.to_numpy(), max_index):
                outliers_list = outliers_list.append(input_series.iloc[max_index])
                input_series.drop(input_series.index[max_index], inplace =True)
            else:
                break
            i += 1
    return outliers_list

def data_read():
    data = pd.read_csv('datasets/Dataset_for_change_detection/Generated_Data_through_experiment/EmptyRCdata.csv')
    data.drop('Days', axis = 1, inplace = True)
    data.fillna(data.mean(), inplace = True)
    return data

def read_sequence(i,data):
    ''' Read from the csv one by one'''
    return data.iloc[i]

def data_plot(points, base, axt, ax_list, num_features, cols, rows, change):
##    ax_list[0].clear()
##    ax_list[0].plot(points[points.columns[0]])
##    print (points[points.columns[0]])
##    axt.plot()
    ax_t[0].clear()
    ax_t[0].plot(base)
    ax_t[1].clear()
    ax_t[1].plot(points)
    ax_t[0].title.set_text('Base Window')
    ax_t[1].title.set_text('Current Window')
    for i in range(num_features):
        ax_list[i].clear()
        ax_list[i].plot(points[points.columns[i]])
        ax_list[i].title.set_text("Feature" + str(i+1))

    if change == 1:
        out = ESD_Test(points.copy(),0.95)
        ax_t[1].plot(out,'r*')
        for i in range(num_features):
            ax_list[i].plot(out[out.columns[i]],'r*')
    
    
# main function
window_size = 40
display = 100
alpha = 0.05
k = 5
offline_num = 100
n = 3000

ind = np.array([i for i in range(window_size)])

data = data_read()
num_features = len(data.columns)

##current_window = data.iloc[:window_size]
current_window = pd.DataFrame()
for i in range(window_size):
    current_window = current_window.append(read_sequence(i,data))
initial_data = data.iloc[:2*window_size]
outliers = ESD_Test(initial_data,0.95)
base_window = initial_data.iloc[:window_size]
##base_window = current_window.copy(deep = True)

change = 0
change_stack = np.array([])
f_change = 0
val = 0
icrl = []
false_alarms = 0
display_data = current_window.copy()

cols = 3
rows = 2
fig = plt.figure(figsize = (12,9))
##ax_list = [fig.add_subplot(cols*rows, (i//rows)+1, (i%cols)+1) \
##           for i in range(num_features)]
##ax_list = [fig.add_subplot(cols,rows,i+1) for i in range(num_features)]
grid = gridspec.GridSpec(2,1, height_ratios = [1,3])
grid_sub_0 = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec = grid[0])
ax_t = [fig.add_subplot(grid_sub_0[0]), fig.add_subplot(grid_sub_0[1])]


ax_list = []
grid_sub_1 = gridspec.GridSpecFromSubplotSpec(cols,rows,subplot_spec = grid[1])
for i in range(num_features):
    ax = fig.add_subplot(grid_sub_1[i])
    ax_list.append(ax)
    
##print (ax_list)
##print (display_data)
data_plot(display_data, base_window, ax_t,ax_list, num_features,cols,rows,change = 0)

##sleep(2)
##print ('2nd plot')
##data_plot(display_data.iloc[2:],ax_list, num_features, cols, rows, change = 0)


def animate(i):

    global window_size,display, alpha, k, offline_num, n, num_features,\
    data, ind, current_window, base_window, change, change_stack, cols, \
    display_data, rows, ax_list, ax_t;

    if i!=0:
        print ("Output", i)

        ind = np.append(ind, window_size+1)[-display:]

        if display_data.shape[0] > display:
            display_data.drop(0, axis = 0, inplace = True)
            display_data.reset_index(drop = True, inplace = True)

        display_data = display_data.append(read_sequence(window_size+i,data))

        current_window.drop(0, axis = 0, inplace = True)
        current_window = current_window.append(read_sequence(window_size + i,data))

        if (change_stack.size >= 2*window_size) and (np.sum(change_stack[-window_size*2:]) == 0):
            base_window = data.iloc[-2*window_size:-window_size]          

        current_window.reset_index(drop = True, inplace = True)
        base_window.reset_index(drop = True, inplace = True)

        change = SPLL(base_window.to_numpy(),current_window.to_numpy(),k)

        change_stack = np.append(change_stack, change)
        if change_stack.size > offline_num:
            change_stack = np.delete(change_stack,0)

        print (display_data.shape)
        data_plot(display_data, base_window, ax_t, ax_list, num_features, cols, rows, change)


ani = animation.FuncAnimation(fig, animate,frames = n + 1,  interval = 1000)
plt.show()
