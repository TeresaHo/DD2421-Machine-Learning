import numpy as np
# Dimensions of Matrices
rows = 7
cols = 4
depth = 5
# Creating matrices
A = np.zeros((rows,cols)) # 2D Matrix of zeros
print(A.shape)

A = np.zeros((depth,rows,cols))  # 3D Matrix of zeros
A = np.ones((rows,cols)) # 2D Matrix of ones
A = np.array([(1,2,3),(4,5,6),(7,8,9)]) # 2D 3x3 matrix with values
# Turn it into a square diagonal matrix with zeros of-diagonal
d = np.diag(A) # Get diagonal as a row vector
print(d)
e = [7,8,9]
print(np.diag(e))
d = np.diag(d) # Turn a row vector into a diagonal matrix
print(d)

def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)
    print(classes)

    prior = np.zeros((Nclasses,1))

    # TODO: compute the values of prior for each class!
    # ==========================
    
    for idx, classname in enumerate(labels):
        prior[classname] += W[idx]
        
    prior /= np.linalg.norm(prior)

    # ==========================

    return prior
prior = np.array([0.33,0.33,0.33])
print(np.linalg.norm(prior))