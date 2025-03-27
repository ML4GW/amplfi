"""Auxiliary functions for distance ansatz see:10.3847/2041-8205/829/1/L15"""

import numpy as np
import scipy as sp


def P(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def Q(x):
    return sp.special.erfc(x / np.sqrt(2)) / 2


def H(x):
    return P(x) / Q(x)


def dHdz(z):
    return -H(z) * (H(z) - z)


def x2(z):
    return z**2 + 1 + z * H(-z)


def x3(z):
    return z**3 + 3 * z + (z**2 + 2) * H(-z)


def x4(z):
    return z**4 + 6 * z**2 + 3 + (z**3 + 5 * z) * H(-z)


def x2_prime(z):
    return 2 * z + H(-z) + z * dHdz(-z)


def x3_prime(z):
    return 3 * z**2 + 3 + 2 * z * H(-z) + (z**2 + 2) * dHdz(-z)


def x4_prime(z):
    return (
        4 * z**3 + 12 * z + (3 * z**2 + 5) * H(-z) + (z**3 + 5 * z) * dHdz(-z)
    )


def f(z, s, m):
    r = 1 + (s / m) ** 2
    r *= x3(z) ** 2
    r -= x2(z) * x4(z)
    return r


def fprime(z, s, m):
    r = 2 * (1 + (s / m) ** 2)
    r *= x3(z) * x3_prime(z)
    r -= x2(z) * x4_prime(z)
    r -= x2_prime(z) * x4(z)
    return r


def moments_from_samples_impl(d):
    """
    Given distance samples, assumed to be posterior
    samples, compute distance moments in monte-carlo sense.

    Args:
        d:
           Distance samples
    """
    d_2 = d**2
    rho = d_2.sum()
    d_3 = d**3
    d_3 = d_3.sum()
    d_4 = d**4
    d_4 = d_4.sum()

    m = d_3 / rho
    s = np.sqrt(d_4 / rho - m**2)
    return rho, m, s


def ansatz_impl(s, m, maxiter=10):
    """Given s and m parameters (see Eqs. 1-3 of
    :doi:`10.3847/0067-0049/226/1/10`) solve for
    mu, sigma, and norm parameters.

    Args:
        s:
            Std. dev calculated from moments
        m:
            Conditional distance mean per pixel
    """
    z0 = m / s
    sol = sp.optimize.root_scalar(
        f, args=(s, m), fprime=fprime, x0=z0, maxiter=maxiter
    )
    if not sol.converged:
        dist_mu = float("inf")
        dist_sigma = 1
        dist_norm = 0
    else:
        z_hat = sol.root
        dist_sigma = m * x2(z_hat) / x3(z_hat)
        dist_mu = dist_sigma * z_hat
        dist_norm = 1 / (Q(-z_hat) * dist_sigma**2 * x2(z_hat))
    return dist_mu, dist_sigma, dist_norm
