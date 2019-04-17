# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    term1 = Y*np.log(P)
    term2 = (1-np.float_(Y))*np.log(1-np.float_(P))
    return -np.sum(term1+term2)