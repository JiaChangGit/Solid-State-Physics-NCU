import numpy as np

#==============================================================================

class FFMatrix:
    """Fourier transformation from real space to k-space."""

    def __init__(self, matrices, shifts):
        for m in matrices[1:]:
            if m.shape != matrices[0].shape:
                raise ValueError('Shape of all matrices should be equal.')
        self._matrices = matrices
        self._shifts = shifts

    #----------------------------------

    @property
    def matrices(self):
        return self._matrices

    @property
    def shifts(self):
        return self._shifts

    #----------------------------------

    def of_kp(self, kp):
        result = np.zeros_like(self.matrices[0], dtype=np.complex128)
        for mat, shift in zip(self.matrices, self.shifts):
            phase = np.exp( 1j * np.dot(shift,kp) )
            result += phase * mat
        return result

#==============================================================================

def kpoints_to_xaxis(kpts):
    """Convert 3-D k-points into a single array with range [0,1].

    Parameters
    ----------

    kpts: array with shape (N,3)
        Coordinates of K-points with shape (N,3), where N is # of k-points.

    Examples
    --------

    >>> a = np.array([[0,0,2],[0,0,3],[0,0,5],[0,0,6]])
    >>> a
    array([[0, 0, 2],
           [0, 0, 3],
           [0, 0, 5],
           [0, 0, 6]])
    >>> kpoints_to_xaxis(a)
    array([0.  , 0.25, 0.75, 1.  ])

    """
    kpts = np.asarray(kpts)
    dArr = np.zeros(len(kpts))
    dArr[1:] = np.linalg.norm( np.diff(kpts, axis=0), axis=1 )
    xArr = np.cumsum(dArr)
    xArr /= max(xArr)
    return xArr

#==============================================================================

def savetxt_table(cols, filename):
    """Save table data to txt.

    Parameters
    ----------

    cols: sequence of 2-elements tuple
        Columns of table. For each tuple, the first element is header neame,
        and the second element is a 1-D array.

    filename: str
        The filename to be saved.

    """
    header = ''
    data = []
    for col in cols:
        header += '{:>14s} '.format(str(col[0]))
        data.append( col[1].reshape(-1,1) )
    np.savetxt(filename, np.hstack(data), header=header, fmt='% 14.7e', comments='')

#==============================================================================

def savetxt_eigvals(kArr, eigvals, filename):
    nbands = eigvals.shape[1]
    data = [
        ('kx',kArr[:,0]),
        ('ky',kArr[:,1]),
        ('kz',kArr[:,2]),
    ]
    for n in range(nbands):
        data.append(('band%s'%(n+1), eigvals[:,n]))
    savetxt_table(data, filename='bands.txt')

#==============================================================================

def reciprocal(av):
    """Compute the reciprocal lattice vector of given lattice vector.

    [b1,b2,b3].transpose == 2*pi*[a1,a2,a3].inverse

    Parameters
    ----------

    av : array-like with shape up to (3,3)

    Returns
    -------

    bv : ndarray with shape (3,3)

    """
    av_nxn = np.array(av)
    bv_nxn = 2 * np.pi * np.linalg.pinv(av_nxn).T
    bv_3x3 = np.zeros((3,3))
    n1, n2 = av_nxn.shape
    bv_3x3[:n1,:n2] = np.array(bv_nxn)
    return bv_3x3

#==============================================================================

def solve_bands(H, kpts):
    """Solve eigenvalues of Hamiltonian H.

    Parameters
    ----------

    H: matrices in dictionary
        The key of H is displacement; The value of H is a NxN matrix.

    kpts: a Nx3 array
        The k-points to be solved.

    Returns
    -------

    evals: a NxM array
        Eigenvalues of Hamiltonian H with shape (N,M), where N is # of k-points
        and M is # of bands.

    """

    # build H of k
    HAll, R = [], []
    for k, v in H.items():
        R.append(k)
        HAll.append(v)
    f = FFMatrix(HAll, R)

    # solve eigenvalues
    evals = []
    for k in kpts:
        Hk = f.of_kp(k)
        Ek = np.linalg.eigvalsh(Hk)
        evals.append(Ek)
    evals = np.asarray(evals)

    return evals

#==============================================================================
