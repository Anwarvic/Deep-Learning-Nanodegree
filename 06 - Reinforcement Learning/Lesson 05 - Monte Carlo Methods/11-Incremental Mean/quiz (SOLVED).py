import numpy as np

def running_mean(x):
    mu = 0
    mean_values = []
    for k in np.arange(0, len(x)):
        # TODO: fill in the update step
        k = k+1
        mu = mu + ((x[k-1]-mu)/k)
        mean_values.append(mu)
    return mean_values