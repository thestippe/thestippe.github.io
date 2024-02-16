import pytensor.tensor as pt
from pytensor.tensor.random.op import RandomVariable
from typing import List, Tuple
from pymc.pytensorf import floatX
from pymc.distributions.distribution import Continuous
import scipy
import numpy as np
from jax import numpy as jnp
import pymc as pm
from pymc.distributions.dist_math import (check_parameters)
from pymc.distributions.shape_utils import rv_size_is_none


class GenParetoRV(RandomVariable):
    # https://en.wikipedia.org/wiki/Generalized_Pareto_distribution
    name: str = "GEV"

    ndim_supp: int = 0

    ndims_params: List[int] = [0, 0, 0]

    dtype: str = "floatX"
    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        xi: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        size: Tuple[int, ...],
    ) -> np.ndarray:
        u = scipy.stats.uniform(loc=0, scale=1).rvs(random_state=rng, size=size)
        return  mu + sigma/xi*(u**(-xi)-1)


class GPD(Continuous):
    rv_op = GenParetoRV()

    def moment(rv, size, xi, mu, sigma):
        mean = pm.math.where(xi<1, mu + sigma/(1-xi), np.inf)
        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean
    
    @classmethod
    def dist(cls, xi, mu, sigma, *args, **kwargs):
        xi = pt.as_tensor_variable(floatX(xi))
        mu = pt.as_tensor_variable(floatX(mu))
        sigma = pt.as_tensor_variable(floatX(sigma))

        return super().dist([xi, mu, sigma], *args, **kwargs)

    def logp(value, xi, mu, sigma):
        scaled = (value-mu)/sigma
        val = pm.math.where(xi, (-(1+1/xi)*pt.log((1+xi*scaled))-pt.log(sigma)), -scaled)
        return check_parameters(
            val,
            sigma > 0,
            scaled>0,
            xi*value+sigma>xi*mu,
            msg="sigma > 0, ((value-mu)/sigma) > 0, xi*value+sigma>xi*mu",
        )