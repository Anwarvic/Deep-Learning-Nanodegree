# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    return np.exp(L)/np.sum(np.exp(L))