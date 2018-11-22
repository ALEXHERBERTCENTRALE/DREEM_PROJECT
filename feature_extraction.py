import numpy as np


def maxAmpOne(list_freq = None, param = None , rep_dim_feature_per_signal = False):
    # Returns the maximum amplitude of a list of frequencies
    if rep_dim_feature_per_signal:
        return 1
    return np.array([max(list_freq)])


def freqMinLimitAmpOne(list_freq = None, param = None , rep_dim_feature_per_signal = False):
    # Returns the (minimum) frequency above which all frequencies have amplitude < amp_lim = param[0]
    if rep_dim_feature_per_signal:
        return 1
    
    amp_lim, = param
    for i in range(len(list_freq)-1,-1,-1):
        if list_freq[i] > amp_lim:
            return np.array([i+1])
    return np.array([0])


def meanOfInterval(signal, freq_min, freq_max):
    return np.average(signal[freq_min:freq_max])
    
def buildIntervals(list_length, interval_width):
    return list([i, i+interval_width] for i in range(0,list_length,interval_width))

def nbPikesOne(list_freq = None, param = None , rep_dim_feature_per_signal = False ):   #param = [interval_width, amp_lim]
    # Returns the number of frequency pikes for a given (amp_lim) amplitude, averaging over an interval
    if rep_dim_feature_per_signal:
        return 1
    interval_width, amp_lim = param
    intervals = buildIntervals(len(list_freq), interval_width)
    nb_pikes = 0
    isIntervalInAPike = [False]
    for minim, maxim in intervals:
        mean_value = meanOfInterval(list_freq, minim, maxim)
        if mean_value > amp_lim:
            isIntervalInAPike.append(True)
        else:
            isIntervalInAPike.append(False)
        if not(isIntervalInAPike[-2]) and isIntervalInAPike[-1]:
            nb_pikes+=1
    return np.array([nb_pikes])





def meanDiffNeighbOne(list_freq = None , param = None , rep_dim_feature_per_signal = False):  #param is useless
    # Returns the average of absolute difference of amplitude between all neighbours frequencies
    if rep_dim_feature_per_signal:
        return 1
    return np.array([np.average(list(abs(list_freq[i+1]-list_freq[i]) for i in range(len(list_freq)-1)))])


    
def stdDeviationNbOne(list_freq = None , param = None , rep_dim_feature_per_signal = False ):  #param = [ n_nb ]
    # Returns the average of standard deviations computed on a given number of points
    if rep_dim_feature_per_signal:
        return 1
    n_nb, = param
    c_max = len(list_freq)//n_nb
    return np.array([np.average(list(np.std(list_freq[c*n_nb:(c+1)*n_nb]) for c in range(c_max)))])

    
def upperRightOne(list_freq = None , param = None , rep_dim_feature_per_signal = False):   #param = [th_amp , th_freq]
    # Returns a boolean, True iff there is a point in the upper right corner, defined by the parameters
    # Consider returning TRUE iff there are more than a given number of points in the upper right corner
    if rep_dim_feature_per_signal:
        return 1
    
    th_amp , th_freq = param
    return np.array([max(list_freq[th_freq:-1])>th_amp])
    


def methodTestOne(list_freq = None, param = None, rep_dim_feature_per_signal = False):
    # Pointless method to test extractFeatureAll
    if rep_dim_feature_per_signal:
        return 2
    
    return np.array([1,2])
    
def extractFeatureAll(h5file_freq , methodOne , param ):
    # Method giving the design matrix of h5file, given a certain method
    # Consider giving a list of methods and returninf the design matrix for all extracted features
    key_list = list(h5file_freq.keys())
    nb_samples = len(h5file_freq[key_list[0]])
    dim_feature_per_signal = methodOne(rep_dim_feature_per_signal = True)
    rep = np.zeros((nb_samples , len(h5file_freq)*dim_feature_per_signal))
    
    for k_id in range(len(key_list)):
        k=key_list[k_id]
        rep[: ,k_id*dim_feature_per_signal:(k_id+1)*dim_feature_per_signal ] =  np.array( list(methodOne(h5file_freq[k][i] , param) for i in range(nb_samples)  ))
    return rep

## to do some testing
# pseudo-h5 file with 3 keys (3 indicators), and 3 samples, of length 29 each
dico = {}
dico["cle1"] = [np.arange(1,30) , np.random.normal(10,5,29) , np.arange(2,31)]
dico["cle2"] = [np.arange(2,31), np.arange(1,30) , np.random.normal(12,6,29)]
dico["cle3"] = [np.random.normal(42,1,29),np.arange(2,31),np.arange(10,39)]
    
    