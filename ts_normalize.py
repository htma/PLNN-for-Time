import numpy as np
def normalize(x):
    """
    Implement a function that normalizes a sequence of a time series.
      - x A numpy array of shape (n,)
    Returns the normalized numpy array. 
    """
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, axis=0, keepdims = True)
    x = x / x_norm

    return x

x = np.array([1,2,3,4])
print(normalize(x))


 

