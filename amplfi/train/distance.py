"""Auxiliary functions for distance ansatz see:10.3847/2041-8205/829/1/L15"""

import torch
from ml4gw.constants import PI


def root_scalar(f, x0, args=(), fprime=None, maxiter=100, xtol=1e-6):
    """
    Find a root of a scalar function.

    Args:
        f (callable): The function whose root is to be found.
        x0 (float): Initial guess.
        args (tuple, optional): Extra arguments passed to the objective
        function `f` and its derivative(s).
        fprime (callable, optional): The derivative of the function.
        xtol (float, optional): The tolerance for the root.
        maxiter (int, optional): The maximum number of iterations.

    Returns:
        dict: A dictionary containing the root, and whether the optimization
        was successful.
    """
    if x0 is None:
        raise ValueError("x0 must be provided")
    res = {"converged": False, "root": None}
    for _ in range(maxiter):
        fx = f(x0, *args)
        if fprime is not None:
            fpx = fprime(x0, *args)
        else:
            fpx = (f(x0 + xtol, *args) - f(x0 - xtol, *args)) / (2 * xtol)
        if abs(fpx) < torch.finfo(torch.float).eps:
            res["root"] = x0
            res["converged"] = True
            return res
        x1 = x0 - fx / fpx
        if abs(x1 - x0) < xtol:
            res["root"] = x1
            res["converged"] = True
            return res
        x0 = x1
    return res


def P(x):
    return torch.exp(-0.5 * x**2) / torch.sqrt(
        torch.tensor(2 * PI, device=x.device)
    )


def Q(x):
    return (
        torch.special.erfc(x / torch.sqrt(torch.tensor(2, device=x.device)))
        / 2
    )


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
        4 * z**3
        + 12 * z
        + (3 * z**2 + 5) * H(-z)
        + (z**3 + 5 * z) * dHdz(-z)
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
    s = torch.sqrt(d_4 / rho - m**2)
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
    sol = root_scalar(f, z0, args=(s, m), fprime=fprime, maxiter=maxiter)
    if not sol["converged"]:
        dist_mu = float("inf")
        dist_sigma = 1
        dist_norm = 0
    else:
        z_hat = sol["root"]
        dist_sigma = m * x2(z_hat) / x3(z_hat)
        dist_mu = dist_sigma * z_hat
        dist_norm = 1 / (Q(-z_hat) * dist_sigma**2 * x2(z_hat))
    return dist_mu, dist_sigma, dist_norm
