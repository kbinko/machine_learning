import numpy as np

def sd(x):
    """Computes the standard deviation of a vector x"""
    return np.sqrt(np.sum((x - np.mean(x))**2) / (len(x) - 1))

def cov(x, y):
    """Computes the covariance between two vectors x and y"""
    return np.sum((x - np.mean(x)) * (y - np.mean(y))) / (len(x) - 1)

def corr(x, y):
    """Computes the correlation between two vectors x and y"""
    return cov(x, y) / (sd(x) * sd(y))

def corr_matrix(X):
    """Computes the correlation matrix for a matrix X"""
    n = X.shape[1]
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            R[i, j] = corr(X[:, i], X[:, j])
    return R

def corr_matrix2(X):
    """Computes the correlation matrix for a matrix X"""
    n = X.shape[1]
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            R[i, j] = corr(X[:, i], X[:, j])
            R[j, i] = R[i, j]
    return R

for i in range(10):
    print(i)
    
if __name__ == "__main__":
    # Test the functions
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    print("sd(x) = ", sd(x))
    print("cov(x, y) = ", cov(x, y))
    print("corr(x, y) = ", corr(x, y))
    X = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    print("corr_matrix(X) = ", corr_matrix(X))
    print("corr_matrix2(X) = ", corr_matrix2(X))
    