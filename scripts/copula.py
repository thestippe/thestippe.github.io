import pytensor.tensor as pt
from pytensor.tensor.random.op import RandomVariable
from typing import List, Tuple
from pymc.pytensorf import floatX
from pymc.distributions.distribution import Continuous
import scipy
from scipy.stats import t, multivariate_t, multivariate_normal
import numpy as np
from jax import numpy as jnp
import pymc as pm
from pymc.distributions.dist_math import (check_parameters)
from pymc.distributions.shape_utils import rv_size_is_none
from pytensor.tensor.linalg import cholesky, det, eigh, solve_triangular, trace
from pytensor.tensor.linalg import inv as matrix_inverse
from pytensor.tensor.random.basic import MvNormalRV, dirichlet, multinomial, multivariate_normal
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.utils import (
    broadcast_params,
    normalize_size_param,
)
from pytensor.tensor.math import betaincinv
from pytensor.tensor.type import TensorType
from scipy import stats
import warnings
import pymc as pm
from functools import partial, reduce
from pytensor.tensor import (
    TensorConstant,
    TensorVariable,
    gammaln,
    get_underlying_scalar_constant_value,
    sigmoid,
)

solve_lower = partial(solve_triangular, lower=True)
solve_upper = partial(solve_triangular, lower=False)


class StudentTCopulaRN(RandomVariable):
    name = "copula"
    signature = "(),(n),(n),(n),(n,n)->(n)"
    dtype = "floatX"
    _print_name = ("StudentTCopula", "\\operatorname{StudentTCopula}")

    @classmethod
    def rng_fn(cls, rng, nu, mu, rho, df, cov, size):
        if size is None:
            # When size is implicit, we need to broadcast parameters correctly,
            # so that the MvNormal draws and the chisquare draws have the same number of batch dimensions.
            # nu broadcasts mu and cov
            if np.ndim(nu) > max(mu.ndim - 1, cov.ndim - 2):
                _, mu, cov = broadcast_params((nu, mu, cov), ndims_params=cls.ndims_params)
            # nu is broadcasted by either mu or cov
            elif np.ndim(nu) < max(mu.ndim - 1, cov.ndim - 2):
                nu, _, _ = broadcast_params((nu, mu, cov), ndims_params=cls.ndims_params)

        mv_samples = multivariate_normal.rng_fn(rng=rng, mean=np.zeros_like(mu), cov=cov, size=size)

        # Take chi2 draws and add an axis of length 1 to the right for correct broadcasting below
        chi2_samples = np.sqrt(rng.chisquare(nu, size=size) / nu)[..., None]

        z = (mv_samples / chi2_samples)
        u = t(df=nu).cdf(z)
        y0 = t(df=df).ppf(u)
        y = mu + rho * y0
        return y


copula = StudentTCopulaRN()

nan_lower_cholesky = partial(cholesky, lower=True, on_error="nan")


def quaddist_matrix(cov=None, chol=None, tau=None, lower=True, *args, **kwargs):
    if len([i for i in [tau, cov, chol] if i is not None]) != 1:
        raise ValueError("Incompatible parameterization. Specify exactly one of tau, cov, or chol.")

    if cov is not None:
        cov = pt.as_tensor_variable(cov)
        if cov.ndim < 2:
            raise ValueError("cov must be at least two dimensional.")
    elif tau is not None:
        tau = pt.as_tensor_variable(tau)
        if tau.ndim < 2:
            raise ValueError("tau must be at least two dimensional.")
        cov = matrix_inverse(tau)
    else:
        chol = pt.as_tensor_variable(chol)
        if chol.ndim < 2:
            raise ValueError("chol must be at least two dimensional.")

        if not lower:
            chol = pt.swapaxes(chol, -1, -2)

        # tag as lower triangular to enable pytensor rewrites of chol(l.l') -> l
        chol.tag.lower_triangular = True
        cov = pt.matmul(chol, pt.swapaxes(chol, -1, -2))

    return cov


def _logdet_from_cholesky(chol: TensorVariable) -> tuple[TensorVariable, TensorVariable]:
    diag = pt.diagonal(chol, axis1=-2, axis2=-1)
    logdet = pt.log(diag).sum(axis=-1)
    posdef = pt.all(diag > 0, axis=-1)
    return logdet, posdef


def quaddist_chol(value, mu, cov):
    """Compute (x - mu).T @ Sigma^-1 @ (x - mu) and the logdet of Sigma."""
    if value.ndim == 0:
        raise ValueError("Value can't be a scalar")
    if value.ndim == 1:
        onedim = True
        value = value[None, :]
    else:
        onedim = False

    chol_cov = nan_lower_cholesky(cov)
    logdet, posdef = _logdet_from_cholesky(chol_cov)

    # solve_triangular will raise if there are nans
    # (which happens if the cholesky fails)
    chol_cov = pt.switch(posdef[..., None, None], chol_cov, 1)

    delta = value - mu
    delta_trans = solve_lower(chol_cov, delta, b_ndim=1)
    quaddist = (delta_trans ** 2).sum(axis=-1)

    if onedim:
        return quaddist[0], logdet, posdef
    else:
        return quaddist, logdet, posdef


class StudentTCopula(Continuous):
    r"""
    Multivariate Student-T log-likelihood.

    .. math::
        f(\mathbf{x}| \nu,\mu,\Sigma) =
        \frac
            {\Gamma\left[(\nu+p)/2\right]}
            {\Gamma(\nu/2)\nu^{p/2}\pi^{p/2}
             \left|{\Sigma}\right|^{1/2}
             \left[
               1+\frac{1}{\nu}
               ({\mathbf x}-{\mu})^T
               {\Sigma}^{-1}({\mathbf x}-{\mu})
             \right]^{-(\nu+p)/2}}

    ========  =============================================
    Support   :math:`x \in \mathbb{R}^p`
    Mean      :math:`\mu` if :math:`\nu > 1` else undefined
    Variance  :math:`\frac{\nu}{\mu-2}\Sigma`
                  if :math:`\nu>2` else undefined
    ========  =============================================

    Parameters
    ----------
    nu : tensor_like of float
        Degrees of freedom, should be a positive scalar.
    Sigma : tensor_like of float, optional
        Scale matrix. Use `scale` in new code.
    mu : tensor_like of float, optional
        Vector of means for the marginals.
    rho : tensor_like of float, optional
        Vector of standard deviations for the marginals.
    df : tensor_like of float
        Degrees of freedom of the marginals, should be a positive scalar.
    scale : tensor_like of float, optional
        The scale matrix.
    tau : tensor_like of float, optional
        The precision matrix.
    chol : tensor_like of float, optional
        The cholesky factor of the scale matrix.
    lower : bool, default=True
        Whether the cholesky fatcor is given as a lower triangular matrix.
    """

    rv_op = copula

    @classmethod
    def dist(cls, nu, *, Sigma=None, mu=0, rho=1, df=1, scale=None, tau=None, chol=None, lower=True, **kwargs):
        cov = kwargs.pop("cov", None)
        if cov is not None:
            warnings.warn(
                "Use the scale argument to specify the scale matrix. "
                "cov will be removed in future versions.",
                FutureWarning,
            )
            scale = cov
        if Sigma is not None:
            if scale is not None:
                raise ValueError("Specify only one of scale and Sigma")
            scale = Sigma
        nu = pt.as_tensor_variable(nu)
        mu = pt.as_tensor_variable(mu)
        rho = pt.as_tensor_variable(rho)
        df = pt.as_tensor_variable(df)
        scale = quaddist_matrix(scale, chol, tau, lower)
        # PyTensor is stricter about the shape of mu, than PyMC used to be
        mu, _ = pt.broadcast_arrays(mu, scale[..., -1])
        rho, _ = pt.broadcast_arrays(rho, scale[..., -1])
        df, _ = pt.broadcast_arrays(df, scale[..., -1])

        return super().dist([nu, mu, rho, df, scale], **kwargs)

    def support_point(rv, size, nu, mu, rho, df, scale):
        # mu is broadcasted to the potential length of scale in `dist`
        mu, _ = pt.random.utils.broadcast_params([mu, nu], ndims_params=[1, 0])
        support_point = mu
        if not rv_size_is_none(size):
            support_point_size = pt.concatenate([size, [mu.shape[-1]]])
            support_point = pt.full(support_point_size, support_point)
        return support_point

    def logp(value, nu, mu, rho, df, scale):
        """
        Calculate logp of Multivariate Student's T distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """

        t = (value - mu) / rho
        sqrt_t2_nu = pt.sqrt(t ** 2 + df)
        z = (t + sqrt_t2_nu) / (2.0 * sqrt_t2_nu)
        x = pt.betainc(df / 2.0, df / 2.0, z)
        v = pt.switch(
            pt.lt(x, 0.5),
            -pt.sqrt(nu) * pt.sqrt((1.0 / betaincinv(nu * 0.5, 0.5, 2.0 * x)) - 1.0),
            pt.sqrt(nu) * pt.sqrt((1.0 / betaincinv(nu * 0.5, 0.5, 2.0 * (1 - x))) - 1.0),
        )
        k = value.shape[-1].astype("floatX")
        mu0 = pm.math.zeros(v.shape[-1])
        quaddist, logdet, ok = quaddist_chol(v, mu0, scale)

        norm = gammaln((nu + k) / 2.0) - gammaln(nu / 2.0) - 0.5 * k * pt.log(nu * np.pi)
        inner = -(nu + k) / 2.0 * pt.log1p(quaddist / nu)
        res = norm + inner - logdet  # pure copula part

        lam = (rho ** -2.0)

        res_t = (
                gammaln((df + 1.0) / 2.0)
                + 0.5 * pt.log(lam / (df * np.pi))
                - gammaln(df / 2.0)
                - (df + 1.0) / 2.0 * pt.log1p(lam * (value - mu) ** 2 / df)
        )

        return check_parameters(res + pt.sum(res_t), ok, nu > 0, msg="posdef, nu > 0")
