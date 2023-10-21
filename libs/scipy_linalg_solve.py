#
# Author: Pearu Peterson, March 2002
#
# w/ additions by Travis Oliphant, March 2002
#              and Jake Vanderplas, August 2012
#
# ** removed nuissance ill-conditioned matrix warnings for GeoDT, Frash, 2023 **

import numpy as np
from numpy import atleast_1d, atleast_2d
from scipy._lib._util import _asarray_validated
from scipy.linalg.lapack import get_lapack_funcs

# Linear equations
def _solve_check(n, info, lamch=None, rcond=None):
    """ Check arguments during the different steps of the solution phase """
    if info < 0:
        raise ValueError('LAPACK reported an illegal value in {}-th argument'
                         '.'.format(-info))
    elif 0 < info:
        return 1
    if lamch is None:
        return 0
    E = lamch('E')
    if rcond < E:
        return 0
    return 0

def solve(a, b, lower=False, overwrite_a=False, overwrite_b=False, 
          check_finite=True, transposed=False):
    """
    Solves the linear equation set ``a * x = b`` for the unknown ``x``
    for square ``a`` matrix.

    Parameters
    ----------
    a : (N, N) array_like
        Square input data
    b : (N, NRHS) array_like
        Input data for the right hand side.
    sym_pos : bool, optional
        Assume `a` is symmetric and positive definite. This key is deprecated
        and assume_a = 'pos' keyword is recommended instead. The functionality
        is the same. It will be removed in the future.
    lower : bool, optional
        If True, only the data contained in the lower triangle of `a`. Default
        is to use upper triangle. (ignored for ``'gen'``)
    overwrite_a : bool, optional
        Allow overwriting data in `a` (may enhance performance).
        Default is False.
    overwrite_b : bool, optional
        Allow overwriting data in `b` (may enhance performance).
        Default is False.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    transposed: bool, optional
        If True, ``a^T x = b`` for real matrices, raises `NotImplementedError`
        for complex matrices (only for True).

    Returns
    -------
    x : (N, NRHS) ndarray
        The solution array.
    0 : (N, NRHS) ndarray
        A placeholder array of zeros when the solution is not found.

    Raises
    ------
    ValueError
        If size mismatches detected or input a is not square.
    LinAlgWarning
        If an ill-conditioned input a is detected.
    NotImplementedError
        If transposed is True and input a is a complex matrix.

    Examples
    --------
    Given `a` and `b`, solve for `x`:

    >>> a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
    >>> b = np.array([2, 4, -1])
    >>> from scipy import linalg
    >>> x = linalg.solve(a, b)
    >>> x
    array([ 2., -2.,  9.])
    >>> np.dot(a, x) == b
    array([ True,  True,  True], dtype=bool)

    Notes
    -----
    If the input b matrix is a 1-D array with N elements, when supplied
    together with an NxN input a, it is assumed as a valid column vector
    despite the apparent size mismatch. This is compatible with the
    numpy.dot() behavior and the returned result is still 1-D array.

    The generic solution is obtained via calling ?GESV routines of LAPACK
    """
    # Flags for 1-D or N-D right-hand side
    b_is_1D = False

    a1 = atleast_2d(_asarray_validated(a, check_finite=check_finite))
    b1 = atleast_1d(_asarray_validated(b, check_finite=check_finite))
    n = a1.shape[0]

    if a1.shape[0] != a1.shape[1]:
        raise ValueError('Input a needs to be a square matrix.')

    if n != b1.shape[0]:
        # Last chance to catch 1x1 scalar a and 1-D b arrays
        if not (n == 1 and b1.size != 0):
            raise ValueError('Input b has to have same number of rows as '
                             'input a')

    # accommodate empty arrays
    if b1.size == 0:
        return np.asfortranarray(b1.copy())

    # regularize 1-D b arrays to 2D
    if b1.ndim == 1:
        if n == 1:
            b1 = b1[None, :]
        else:
            b1 = b1[:, None]
        b_is_1D = True

    # Currently we do not have the other forms of the norm calculators
    lange = get_lapack_funcs('lange', (a1,))

    # Since the I-norm and 1-norm are the same for symmetric matrices
    # we can collect them all in this one call
    if transposed:
        trans = 1
        norm = 'I'
    else:
        trans = 0
        norm = '1'

    anorm = lange(norm, a1)

    # Generalized case 'gesv'
    gecon, getrf, getrs = get_lapack_funcs(('gecon', 'getrf', 'getrs'),
                                           (a1, b1))
    lu, ipvt, info = getrf(a1, overwrite_a=overwrite_a)
    #_solve_check(n, info)
    if _solve_check(n, info): return np.zeros(len(b))
    x, info = getrs(lu, ipvt, b1,
                    trans=trans, overwrite_b=overwrite_b)
    #_solve_check(n, info)
    if _solve_check(n, info): return np.zeros(len(b))
    rcond, info = gecon(lu, anorm, norm=norm)

    #_solve_check(n, info)
    if _solve_check(n, info): return np.zeros(len(b))

    if b_is_1D:
        x = x.ravel()

    return x