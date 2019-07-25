import numpy as np

# This is the sequence (corresponding to successively sampled returns). 
# Feel free to change it!
x = np.hstack((np.ones(10), 10*np.ones(10)))

# These are the different step sizes alpha that we will test.  
# Feel free to change it!
alpha_values = np.arange(0,.3,.01)+.01

#########################################################
# Please do not change any of the code below this line. #
#########################################################

def running_mean(x):
    mu = 0
    mean_values = []
    for k in np.arange(0, len(x)):
        mu = mu + (1.0/(k+1))*(x[k] - mu)
        mean_values.append(mu)
    return mean_values
    
def forgetful_mean(x, alpha):
    mu = 0
    mean_values = []
    for k in np.arange(0, len(x)):
        mu = mu + alpha*(x[k] - mu)
        mean_values.append(mu)
    return mean_values

def print_results():
    """
    prints the mean of the sequence "x" (as calculated by the
    running_mean function), along with analogous results for each value of alpha 
    in "alpha_values" (as calculated by the forgetful_mean function).
    """
    print('The running_mean function returns:', running_mean(x)[-1])
    print('The forgetful_mean function returns:')
    for alpha in alpha_values:
        print(np.round(forgetful_mean(x, alpha)[-1],4), \
        '(alpha={})'.format(np.round(alpha,2)))