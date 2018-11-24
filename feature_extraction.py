import numpy as np
import pickle

## Auxiliary functions
def meanOfInterval(signal, freq_min, freq_max):
    return np.average(signal[freq_min:freq_max])
    
def buildIntervals(list_length, interval_width):
    return list([i, i+interval_width] for i in range(0,list_length,interval_width))

def mobilMean(signal , interval_width):  #use an odd number as interval_width, otherwise the width will be interval_width + 1.
    # side_interval = interval_width//2
    # return list( np.average( signal[ max(i-side_interval , 0) : min( i+side_interval+1 , len(signal))]) for i in range(len(signal)) )
    cumsum = np.cumsum(np.insert(signal, 0, 0)) 
    return (cumsum[interval_width:] - cumsum[:-interval_width]) / float(interval_width)


## methodOne(s) : feature-extraction methods

def maxAmpOne(list_freq = None, param = None , rep_dim_feature_per_signal = False):  # param useless
    # Returns the maximum amplitude of a list of frequencies
    if rep_dim_feature_per_signal:
        return 1
    return [max(list_freq)]


def freqMinLimitAmpOne(list_freq = None, param = None , rep_dim_feature_per_signal = False):  # param = [amp_lim]
    # Returns the (minimum) frequency above which all frequencies have amplitude < amp_lim = param[0]
    if rep_dim_feature_per_signal:
        return 1
    
    amp_lim, = param
    for i in range(len(list_freq)-1,-1,-1):
        if list_freq[i] > amp_lim:
            return [i+1]
    return [0]



def nbPikesOne(list_freq = None, param = None , rep_dim_feature_per_signal = False ):   #param = [interval_width, amp_lim]
    # Returns the number of frequency pikes for a given (amp_lim) amplitude, averaging with a mobile mean
    if rep_dim_feature_per_signal:
        return 1
    interval_width, amp_lim = param
    mobil_mean_list_freq = mobilMean(list_freq , interval_width)
    nb_pikes = 0
    # is_interval_in_a_pike = [False , False]
    # for i in range(len(mobil_mean_list_freq)):
    #     if mobil_mean_list_freq[i] > amp_lim:
    #         is_interval_in_a_pike = [True , is_interval_in_a_pike[0] ]
    #     else:
    #         is_interval_in_a_pike = [False , is_interval_in_a_pike[0] ]
    #     if not(is_interval_in_a_pike[1]) and is_interval_in_a_pike[0]:
    #         nb_pikes+=1
    
    is_interval_in_a_pike = 4   #code : 1 = [T,T] , 2 = [T,F] , 3 = [F,T] , 4 = [F,F]
    for i in range(len(mobil_mean_list_freq)):
        if mobil_mean_list_freq[i] > amp_lim:
            is_interval_in_a_pike = 1 if is_interval_in_a_pike in [1,2] else 2
        else:
            is_interval_in_a_pike = 3 if is_interval_in_a_pike in [1,2] else 4
        if is_interval_in_a_pike==2:
            nb_pikes+=1
    return [nb_pikes]


def nbPikesFastOne(list_freq = None, param = None , rep_dim_feature_per_signal = False ):   #param = [interval_width, amp_lim]
    # Returns the number of frequency pikes for a given (amp_lim) amplitude, averaging over an interval
    if rep_dim_feature_per_signal:
        return 1
    interval_width, amp_lim = param
    intervals = buildIntervals(len(list_freq), interval_width)
    nb_pikes = 0
    is_interval_in_a_pike = [False , False]
    # for minim, maxim in intervals:
    #     mean_value = meanOfInterval(list_freq, minim, maxim)
    #     if mean_value > amp_lim:
    #         is_interval_in_a_pike = [True , is_interval_in_a_pike[0] ]
    #     else:
    #         is_interval_in_a_pike = [False , is_interval_in_a_pike[0] ]
    #     if not(is_interval_in_a_pike[1]) and is_interval_in_a_pike[0]:
    #         nb_pikes+=1
    
    is_interval_in_a_pike = 4   #code : 1 = [T,T] , 2 = [T,F] , 3 = [F,T] , 4 = [F,F]
    for minim, maxim in intervals:
        mean_value = meanOfInterval(list_freq, minim, maxim)
        if mean_value > amp_lim:
            is_interval_in_a_pike = 1 if is_interval_in_a_pike in [1,2] else 2
        else:
            is_interval_in_a_pike = 3 if is_interval_in_a_pike in [1,2] else 4
        if is_interval_in_a_pike==2:
            nb_pikes+=1
    return [nb_pikes]


def indexMaxAmpOne(list_freq = None , param = None , rep_dim_feature_per_signal = False):  # param = [ interval_width ]
    # Give the index of maximum amplitude, for the data averaged with a mobile mean of size interval_width
    if rep_dim_feature_per_signal:
        return 1
    interval_width, = param
    mobil_mean_list_freq = mobilMean(list_freq , interval_width)
    return [mobil_mean_list_freq.index(max(mobil_mean_list_freq))]

def indexMaxAmpFastOne(list_freq = None , param = None , rep_dim_feature_per_signal = False):  # param = [ interval_width ]
    # Give the index of maximum amplitude, for the data averaged over fixed intervals. Probably way faster than indexMaxAmpOne, though less accurate.
    if rep_dim_feature_per_signal:
        return 1
    interval_width, = param
    i_max = len(list_freq)//interval_width
    #side_interval = interval_width//2
    list_mean = list( np.average( list_freq[i*interval_width:(i+1)*interval_width]) for i in range(i_max) )
    return [ list_mean.index(max(list_mean))*interval_width + interval_width//2 ]


def meanDiffNeighbOne(list_freq = None , param = None , rep_dim_feature_per_signal = False):  #param is useless
    # Returns the average of absolute difference of amplitude between all neighbours frequencies
    if rep_dim_feature_per_signal:
        return 1
    return [np.average(list(abs(list_freq[i+1]-list_freq[i]) for i in range(len(list_freq)-1)))]


    
def stdDeviationNbOne(list_freq = None , param = None , rep_dim_feature_per_signal = False ):  #param = [ n_nb ]
    # Returns the average of standard deviations computed on a given number of points (separation of x-axis in intervals of the same length)
    if rep_dim_feature_per_signal:
        return 1
    n_nb, = param
    c_max = len(list_freq)//n_nb
    return [np.average(list(np.std(list_freq[c*n_nb:(c+1)*n_nb]) for c in range(c_max)))]

    
def upperRightOne(list_freq = None , param = None , rep_dim_feature_per_signal = False):   # param = [th_amp , th_freq]
    # Returns a boolean, True iff there is a point in the upper right corner, defined by the parameters
    # Consider returning TRUE iff there are more than a given number of points in the upper right corner
    if rep_dim_feature_per_signal:
        return 1
    
    th_amp , th_freq = param
    return [max(list_freq[th_freq:-1])>th_amp]
    


def methodTestOne(list_freq = None, param = None, rep_dim_feature_per_signal = False):  # param useless
    # Pointless method to test extractFeatureAll
    if rep_dim_feature_per_signal:
        return 2
    
    return np.array([1,2])

## Global feature-extraction methods
    
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

def extractMultiFeatureAll(h5file_freq , list_methodOne , list_param , save = False , name_save = None):
    # Returns the concatenation of design matrices for a list of methods
    nb_samples = len(h5file_freq[list(h5file_freq.keys())[0]])
    sum_dim_feature_per_signal = sum( methodOne(rep_dim_feature_per_signal = True) for methodOne in list_methodOne )
    rep = np.zeros((nb_samples , len(h5file_freq)*sum_dim_feature_per_signal ))
    
    c = 0
    i = 0
    for methodOne in list_methodOne :
        temp = len(h5file_freq)*methodOne(rep_dim_feature_per_signal = True)
        rep[:,c:c+temp] = extractFeatureAll(h5file_freq , methodOne , list_param[i] )
        i+=1
        c+=temp
    
    if save:
        temp_var_file = open(name_save + '.txt','wb')
        pickle.dump(rep , temp_var_file)
        temp_var_file.close()
        
        #Use next 3 lines to read
        # temp_var_file = open(name_save + '.txt','rb')
        # rep = pickle.load(temp_var_file)
        # temp_var_file.close()
        
        #np.savetxt( name_save + '.txt' , rep , delimiter=',', fmt="%s")
    
    return rep

## to do some testing
# pseudo-h5 file with 3 keys (3 indicators), and 3 samples, of length 29 each
dico = {}
dico["cle1"] = [np.arange(1,30) , np.random.normal(10,5,29) , np.arange(2,31) , np.random.uniform(12,45,29) ]
dico["cle2"] = [np.arange(2,31), np.arange(1,30) , np.random.uniform(12,45,29) , np.random.normal(12,6,29)]
dico["cle3"] = [np.random.normal(42,1,29), np.random.uniform(12,45,29), np.arange(2,31),np.arange(10,39)]
    
    