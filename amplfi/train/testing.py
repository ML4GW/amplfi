from pathlib import Path

import bilby
import healpy as hp
import matplotlib.pyplot as plt
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


class Result(bilby.result.Result):
    def get_sky_projection(self, nside: int):
        """Store a HEALPix array with the sky coordinates

        Args:
            nside: nside parameter for healpy
        """
        ra = self.posterior["phi"]
        dec = self.posterior["dec"]
        theta = np.pi / 2 - dec

        num_samples = len(ra)

        # mask out non physical samples;
        mask = (ra > -np.pi) * (ra < np.pi)
        mask &= (theta > 0) * (theta < np.pi)

        ra = ra[mask]
        dec = dec[mask]
        theta = theta[mask]

        # calculate number of samples in each pixel
        NPIX = hp.nside2npix(nside)
        ipix = hp.ang2pix(nside, theta, ra)
        ipix = np.sort(ipix)
        uniq, counts = np.unique(ipix, return_counts=True)

        # create empty map and then fill in non-zero pix with density
        # estimated by fraction of total samples in each pixel
        m = np.zeros(NPIX)
        m[np.in1d(range(NPIX), uniq)] = counts / num_samples

        return m

    def calculate_searched_area(self, nside: int):
        healpix = self.get_sky_projection(nside)

        ra_inj = self.injection_parameters["phi"]
        dec_inj = self.injection_parameters["dec"]
        theta_inj = np.pi / 2 - dec_inj
        true_ipix = hp.ang2pix(nside, theta_inj, ra_inj)

        sorted_idxs = np.argsort(healpix)[
            ::-1
        ]  # sort pixels in descending order
        # count number of pixels before hitting the pixel with injection
        # in the sorted array
        num_pix_before_injection = 1 + np.argmax(sorted_idxs == true_ipix)
        searched_area = num_pix_before_injection * hp.nside2pixarea(
            nside, degrees=True
        )
        return searched_area

    def plot_mollview(self, nside: int, outpath: Path = None):
        healpix = self.get_sky_projection(nside)
        ra_inj = self.injection_parameters["phi"]
        dec_inj = self.injection_parameters["dec"]
        theta_inj = np.pi / 2 - dec_inj
        plt.close()
        # plot molleweide
        fig = hp.mollview(healpix)
        hp.visufunc.projscatter(
            theta_inj, ra_inj, marker="x", color="red", s=150
        )

        plt.savefig(outpath)

        return fig

    def get_dist_params(self):
        """Get d^2, d^3, d^4 moments from posterior samples.
        Note that this is not conditioned per pixel."""
        d = self.posterior["distance"]
        # calculate moments
        d_2 = d**2
        rho = d_2.sum()
        d_3 = d**3
        d_3 = d_3.sum()
        d_4 = d**4
        d_4 = d_4.sum()

        m = d_3 / rho
        s = np.sqrt(d_4 / rho - m**2)
        return rho, m, s

    def calculate_distance_ansatz(self, maxiter=10):
        """Calculate the DISTMU, DISTSIGMA, DISTNORM parameters"""
        rho, m, s = self.get_dist_params()
        z0 = m / s
        sol = sp.optimize.root_scalar(f, args=(s, m), fprime=fprime, x0=z0)
        if not sol.converged:
            self.dist_mu = 0
            self.dist_sigma = float("inf")
            self.dist_norm = 0
            return
        z_hat = sol.root
        sigma = m * x2(z_hat) / x3(z_hat)
        mu = sigma * z_hat
        N = 1 / (Q(-z_hat) * sigma**2 * x2(z_hat))
        self.dist_mu = mu
        self.dist_sigma = sigma
        self.norm = N
