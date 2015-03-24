__all__ = ['mvee']
import math
import numpy as np
try:
    from scipy.spatial import ConvexHull
except ImportError:
    def _getConvexHull(points):
        return points
else:
    def _getConvexHull(points):
        hull = ConvexHull(points)
        return points[np.unique(hull.simplices)]

def mvee(points, tol=1.e-4, limits=10000):
    """
    Finds the minimum volume enclosing ellipsoid (MVEE) of a set of data points
    in the M-dimentional space.

    Parameters
    ----------
    points : (N, M) array_like
        A array of N points in the M-dimentional space. N must be larger than M.
    tol : float, optional
        Error in the solution with respect to the optimal value.
    limits : int, optional
        Maximal number of iteration.

    Returns
    -------
    A : (M,M) ndarray
        The matrix of the ellipse equation in the 'center form':
        (x-c)^{T} A^{-1} (x-c) = 1,
        where the eigenvalues of A are the squares of the semiaxes.        
    c : (M,) ndarray
        The center of the ellipse.

    Notes
    -----
    This function is ported from the MATLAB routine 
    ``Minimum Volume Enclosing Ellipsoid'' (see [1]_ and [2]_)
    by Nima Moshtagh (nima@seas.upenn.edu) at University of Pennsylvania.
    Note that the output matrix A here is different from the original MATLAB 
    routine, where it returns A^{-1} instead.

    References
    ----------
    .. [1] http://www.mathworks.com/matlabcentral/fileexchange/9542-minimum-volume-enclosing-ellipsoid/content/MinVolEllipse.m
    .. [2] http://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python

"""    
    P = _getConvexHull(points)
    N, d = P.shape
    if N <= d:
        raise ValueError("The number of points must be larger than the number of dimensions.")
    dp1_inv = 1./float(d+1)
    Q = np.vstack((P.T, np.ones(N)))
    err = tol + 1.
    u = np.ones(N)/float(N)
    while err > tol and limits > 0:
        X_inv = np.linalg.inv(np.einsum('ij,j,kj', Q, u, Q))
        M = np.einsum('ji,jk,ki->i', Q, X_inv, Q)
        j = np.argmax(M)
        step_size = (1.-d/(M[j]-1.))*dp1_inv
        u[j] -= 1.
        err = math.sqrt((u*u).sum())*math.fabs(step_size)
        u *= (1.-step_size)
        u[j] += 1.
        u /= u.sum()
        limits -= 1
    c = np.dot(u, P)
    A = (np.einsum('ji,j,jk', P, u, P) - np.outer(c,c)) * float(d)
    return A, c

