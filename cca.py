# Author: James Jensen <jdjensen@eng.ucsd.edu>
#
# License: BSD 3 clause

import warnings
import operator

import scipy
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.linalg import norm
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
from sklearn.linear_model import LinearRegression
from scipy.sparse import issparse
from functools import reduce
MACHINE_PRECISION = np.finfo(np.float).eps

# Todo: if correlation is negative, multiply one set of coefficients by -1
# Todo: profile and speed up

###############################################################################


class BaseCCA(BaseEstimator, RegressorMixin):

    def loss(self, x, y, null_is_nan=False):
        # Todo: make sure this is accessible to GridSearchCV. May be more complicated than I expected.
        a, b = self.coef_[-1]
        if np.all(a == 0) or np.all(b == 0):
            warnings.warn("No nonzero coefficients of {}")
            mse = 4  # maximum mse between unit vectors. e.g. [1] and [-1]
            if null_is_nan:
                mse = np.nan
        else:
            x = self.validate_input(x)
            y = self.validate_input(y)
            transformed_x = np.dot(x, a)
            transformed_y = np.dot(y, b)
            # scaling makes the MSEs comparable and bounded by 4
            transformed_x = make_unit_vector(transformed_x)
            transformed_y = make_unit_vector(transformed_y)
            mse = mean_squared_error(transformed_x, transformed_y)
        return mse

    def score(self, x, y):
        mse = self.loss(x, y)
        return 1 - (mse / 4.0)

    def validate_input(self, x):
        if issparse(x):
            x = x.todense()
        x = np.array(x)
        if x.ndim == 1:
            x = np.atleast_2d(x).T
        if not np.issubdtype(x.dtype, np.float) and not np.issubdtype(x.dtype, np.integer):
            try:
                x = x.astype(np.float64)
            except Exception as e:
                raise TypeError('Could not cast input array to float dtype: argument must be a string or a number')
                #raise RuntimeError(" {}".format(e.message))
        #print type(x)
        if np.isnan(x).any():
            raise ValueError("nan found in input array")
        if np.isinf(x).any():
            raise ValueError("inf or -inf found in input array")
        x = preprocessing.scale(x, with_mean=self.with_mean, with_std=self.with_std)
        return x


class CCA(BaseCCA):
    def __init__(self,method='qr_svd'):
        self.method = method
        self.corr_ = None
        self.coef_ = None
        # Todo: iterative fitting, appending results, etc
        # Todo: transform(), refit()

    def fit(self, x, y, k=1, method=None):
        method = self.method if method is None else method
        if method == 'qr_svd':
            corr, a, b = qr_svd_cca(x, y, k)
        elif method == 'covar_eigen':
            corr, a, b = covar_eigen_cca(x, y, k)
        else:
            raise ValueError('{} is not a supported CCA method'.format(method))
        self.corr_ = corr
        self.coef_ = [a, b]
        return self


def qr_svd_cca(x, y, k=1, right_singular_vector=None):
    """
    #Todo: does not seem to be working for k>1
    # advantage: seems more memory-efficient than covar eigen
    # but can't deal with p > n?
    :param x:
    :param y:
    :param k:
    :param right_singular_vector:
    :param scale_coeffs:
    :return:
    """
    # for qr() performance overwrite_a=True, check_finite=False
    # svds can take a warm start right singular vector
    # I can also decrease precision with tol=something less than machine precision
    q1, r1 = scipy.linalg.qr(x, mode='economic')  # r is upper triangular
    q2, r2 = scipy.linalg.qr(y, mode='economic')
    q = q1.T.dot(q2)
    u, s, vt = scipy.sparse.linalg.svds(q, k=k, which="LM", v0=right_singular_vector)
    #print(u)
    #print(s)
    #print(vt)
    sorted_order = np.argsort(s)[::-1]
    a = scipy.linalg.solve_triangular(r1, u)
    b = scipy.linalg.solve_triangular(r2, vt.T)
    #a = rescale(a, x)
    #a = make_unit_vector(a)
    #b = rescale(b, y)
    #b = make_unit_vector(b)
    return s[sorted_order], a[:, sorted_order], b[:, sorted_order]


def check_and_squeeze_array(a):
    if isinstance(a, np.matrixlib.defmatrix.matrix):
        a = np.array(a)
    if a.ndim > 1:
        a = a.squeeze()
    return a


def compute_scaled_eigvec_pair(eigvec, cyy_i, cyx):
    eigvec = np.real(eigvec)
    if eigvec.ndim > 1:
        eigvec = eigvec.squeeze()
    other_eigvec = reduce(np.dot, [cyy_i, cyx, eigvec])
    x_coeffs = make_unit_vector(eigvec)
    y_coeffs = make_unit_vector(other_eigvec)
    x_coeffs = check_and_squeeze_array(x_coeffs)
    y_coeffs = check_and_squeeze_array(y_coeffs)
    return x_coeffs, y_coeffs


def cancorr_eigendecompose(cxx_i, cxy, cyy_i, cyx, k=1, x_first=True, right_eigvec_guess=None):
    xprod = np.dot(cxx_i, cxy)
    yprod = np.dot(cyy_i, cyx)
    m = np.dot(xprod, yprod) if x_first else np.dot(yprod, xprod)
    eigvals, eigvecs = scipy.sparse.linalg.eigs(m, k=k, return_eigenvectors=True, which='LM',
                                              v0=right_eigvec_guess)
    #eigval, eigvec = scipy.linalg.eig(m, left=True, right=False)
    #print('Eigvals:', np.real(eigvals))
    eigvals = np.real(eigvals)
    # has some convergence issues for returning eigenvectors...
    # eigvals are not in sorted order, either
    ccs = np.sqrt(eigvals)
    #print(ccs)
    for cc in ccs:
        for op, val, name in [(operator.gt, 1.0, 'greater'), (operator.lt, 0.0, 'less')]:
            if op(cc, val):
                raise ArithmeticError(
                "Canonical correlation of {:.10} is {} than {} with Arnoldi iteration".format(cc, name, val))
    eigvec_pair_args = [cyy_i, cyx] if x_first else [cxx_i, cxy]
    x_coeffs, y_coeffs = compute_scaled_eigvec_pair(eigvecs, *eigvec_pair_args)
    return ccs, x_coeffs, y_coeffs


def covar_eigen_cca(x, y, k=1):
    # Todo: fix support for k>1
    n, px = x.shape
    xy = np.hstack([x, y])
    xy = preprocessing.scale(xy)
    full_covar = xy.T.dot(xy) / (n - 1)
    cxx = full_covar[:px, :px]
    cxx_i = sym_inv_or_pinv(cxx)
    cyy = full_covar[px:, px:]
    cyy_i = sym_inv_or_pinv(cyy)
    cxy = full_covar[:px, px:]
    cyx = full_covar[px:, :px]
    corr, x_coeffs, y_coeffs = cancorr_eigendecompose(cxx_i, cxy, cyy_i, cyx, k=k)
    return corr, x_coeffs, y_coeffs


def sym_inv_or_pinv(symmetric_matrix):
    """
    We don't care about singularity, since we don't need a unique solution
    Note that pinvh is only valid for symmetric matrices
    """
    try:
        inv = scipy.linalg.inv(symmetric_matrix)  # is this safe? does it help?
    except scipy.linalg.LinAlgError as e:
        #inv = scipy.linalg.pinvh(symmetric_matrix)
        inv = scipy.linalg.pinv(symmetric_matrix)
    return inv


def make_unit_vector(a):
    """

    :param a:
    :return:
    """
    nm = scipy.linalg.norm(a)
    return np.divide(a, nm)

########################################################################################################################
# Iterative alternating regression

def deflate(xp, x, w):
    """

    :param xp:
    :param x:
    :param w:
    :return:
    """
    qx = xp.T.dot(x).dot(w)
    qx = make_unit_vector(qx)
    xp = xp - xp.dot(np.outer(qx, qx))
    return xp

def rescale(coeffs, mat):
    """

    :param coeffs:
    :param mat:
    :return:
    """
    dot_prod = np.dot(mat, coeffs)
    nm = norm(dot_prod)  #l2 or euclidean norm
    if nm != 0:
        scaled_coeffs = np.divide(coeffs, nm)
    else:
        warnings.warn("zero norm in coefficient scaling")
        scaled_coeffs = coeffs
        # Todo: what does this mean and how should I respond?
    return scaled_coeffs


class ALSCCA(BaseCCA):
    """Canonical correlation by iterative alternating least squares.

    While canonical correlation analysis (CCA) can be done by matrix
    decomposition or partial least squares, it can also be done by iterative
    alternating least squares.

    [math here]


    This method of estimating canonical correlation is flexible, allowing a
    different regression estimator to be chosen for each of the datasets, and
    virtually any kind of regression estimator may be used.
    This allows use of various regularizations, as well as other constraints
    such as non-negativity.

    Subsequent quasi-orthogonal component pairs can be obtained by calling the
    ``fit()`` function repeatedly.

    Parameters
    ----------
    estimator_1, estimator_2 : parameterized regression estimators.
        These should inherit from LinearModel and RegressorMixin.

    center : boolean, optional, default True
        If ``True``, both datasets will be centered before fitting, such that
        all features have mean zero.

    normalize : boolean, optional, default True
        If ``True``, both datasets will be normalized before fitting, such that
        all features have unit variance.

    max_iter: int, optional, default 30
        The maximum number of iterations of alternating least squares.

    tol: float, optional
        The tolerance for the optimization. If the improvement in the objective
        function from one iteration to the next is less than ``tol``, the
        fitting terminates.

    n_restarts: int, optional, default 10
        The number of random restarts. Since solving CCA by alternating least
        squares is not convex but biconvex, random restarts are done to avoid
        local optima.

    parallel_restarts : boolean, optional, default True
        If ``True``, each restart will be performed on a different thread.

    random_state: int or None, optional
        Seed for random number generator.

    Attributes
    ----------

    ``corr_`` : array, shape = (n_components,)
        canonical correlation for each successive canonical component
    ``coef_`` : array, shape = (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)

    Notes
    -----

    """

    def __init__(self, estimator_1=None, estimator_2=None, n_iter=30, n_restarts=10,
                 tol=1E-5, verbose=False, random_state=None,
                 with_mean=True, with_std=True):
        self.estimator_1 = estimator_1
        self.estimator_2 = estimator_2
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.tol = tol
        self.corr_ = None
        self.unscaled_coef_ = None
        self.coef_ = None
        self.random_state = random_state
        self.with_mean = with_mean
        self.with_std = with_std
        self.verbose = verbose

    def refit(self, x, y, k=1):
        # deflate based on previous
        # store current results
        # then fit
        # then merge past and new results

        if self.corr_ is None:
            return self.fit(x, y, k=k)
        else:
            prev_corr = self.corr_
            prev_coef = self.coef_
            prev_unscaled_coef_ = self.unscaled_coef_
            unscaled_as, unscaled_bs = [np.hstack(coef_list_of_lists) for coef_list_of_lists in zip(*prev_unscaled_coef_)]
            deflated_x = deflate(x, x, unscaled_as)
            deflated_y = deflate(y, y, unscaled_bs)
            self.fit(deflated_x, deflated_y, k=k)
            self.corr_ = prev_corr + self.corr_
            self.coef_ = prev_coef + self.coef_
            self.unscaled_coef_ = prev_unscaled_coef_ + self.unscaled_coef_
            return self

    def fit(self, x, y, k=1):
        # Todo: support for parallelization of the restarts
        """Fit model by iterative alternating least squares

        Parameters
        -----------
        x : ndarray or scipy.sparse matrix, (n_samples, n_features_1)
            First dataset

        y : ndarray or scipy.sparse matrix, (n_samples, n_features_2)
            Second dataset
        k : number of components (not greater than min(n_samples, n_features_1, n_features_2))

        Notes
        -----

        """
        x = self.validate_input(x)
        y = self.validate_input(y)
        estimator_1, estimator_2 = [est if est is not None else LinearRegression() for est in [self.estimator_1, self.estimator_2]]
        random_state = check_random_state(self.random_state)
        self.corr_ = []
        self.coef_ = []
        self.unscaled_coef_ = []
        for i in range(k):
            deflated_x = x if i == 0 else deflate(deflated_x, x, self.unscaled_coef_[-1][0].T)
            deflated_y = y if i == 0 else deflate(deflated_y, y, self.unscaled_coef_[-1][1].T)
            best_corr = 0
            best_a = np.zeros((x.shape[1], 1))
            best_b = np.zeros((y.shape[1], 1))
            for j in range(0, self.n_restarts):
                a, b = als_cca(deflated_x, deflated_y, estimator_1, estimator_2, orig_x=x, orig_y=y,
                               n_iter=self.n_iter, tol=self.tol, random_state=random_state)
                if np.all(a == 0) or np.all(b == 0):
                    corr = 0
                    warnings.warn("All coefficients zero: regularization is too strong")
                else:
                    corr, pval = scipy.stats.pearsonr(np.dot(deflated_x, a.T), np.dot(deflated_y, b.T))
                    if corr < 0:
                        corr *= -1
                        a *= -1
                if corr > best_corr:
                    best_corr, best_a, best_b = corr, a, b
            if best_corr == 0:
                scaled_coeffs = [best_a, best_b]
            else:
                scaled_coeffs = [np.squeeze(make_unit_vector(best)) for best in [best_a, best_b]]
            self.corr_.append(best_corr)
            self.coef_.append(scaled_coeffs)
            self.unscaled_coef_.append((best_a, best_b))
        return self

    def predict(self, x):
        # Doesn't make sense to have a predict method, right?
        x = self.validate_input(x)
        a = self.coef_[-1][0]
        return np.dot(x, a.T)


def als_cca(x, y, estimator_1, estimator_2, n_iter=30, tol=1E-5, orig_x=None, orig_y=None, a=None, b=None, random_state=None):
    """

    """
    orig_x = x if orig_x is None else orig_x
    orig_y = y if orig_y is None else orig_y
    ra = random_unit_vector(x.shape[1], is_nonneg_estimator(estimator_1), random_state)
    rb = random_unit_vector(y.shape[1], is_nonneg_estimator(estimator_2), random_state)
    a = rescale(ra, x) if a is None else a
    b = rescale(rb, y) if b is None else b
    index = 0
    obj = float('-inf')
    while index < n_iter:
        transformed_x = np.dot(x, a)
        transformed_y = np.dot(y, b)
        new_obj = mean_squared_error(transformed_x, transformed_y)

        if np.isclose(new_obj, 0.0):
            break
        obj_function_dif = abs(new_obj - obj) / new_obj

        obj = new_obj
        if obj_function_dif < tol:
            break
        estimator_1.fit(x, np.ravel(transformed_y))
        a = estimator_1.coef_.reshape(a.shape)
        a = rescale(a, x)
        estimator_2.fit(y, np.ravel(transformed_x))
        b = estimator_2.coef_.reshape(b.shape)
        b = rescale(b, y)
        index += 1
    if not np.all(a == 0):
        a = rescale(a, orig_x)
    if not np.all(b == 0):
        b = rescale(b, orig_y)
    # Todo: verify Sigg does it by the original data now
    return a.T, b.T


def is_nonneg_estimator(estimator):
    pos = False
    if hasattr(estimator, 'positive'):
        if estimator.positive:
            pos = True
    return pos


def random_unit_vector(length, pos=False, random_state=None):
    """

    """
    if random_state is None:
        random_state = np.random.RandomState()
    #vector = random_state.standard_normal(size=length).reshape((length, 1))
    vector = random_state.uniform(size=length).reshape((length, 1))
    if pos:
        vector = np.abs(vector)
    unit_vector = make_unit_vector(vector)
    return unit_vector
