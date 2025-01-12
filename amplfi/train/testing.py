from pathlib import Path

import bilby
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from astropy import io, table
from astropy import units as u

"""Auxiliary functions for distance ansatz see:10.3847/2041-8205/829/1/L15"""


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


def dist_moments_from_samples_impl(d):
    # calculate moments and evaluate rho, m, s
    d_2 = d**2
    rho = d_2.sum()
    d_3 = d**3
    d_3 = d_3.sum()
    d_4 = d**4
    d_4 = d_4.sum()

    m = d_3 / rho
    s = np.sqrt(d_4 / rho - m**2)
    return rho, m, s


def distance_ansatz_impl(s, m, maxiter=10):
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


def nest2uniq(nside, ipix):
    return 4 * nside * nside + ipix


class Result(bilby.result.Result):
    def get_sky_projection(self, nside: int):
        """Store a HEALPix array with the sky coordinates

        Args:
            nside: nside parameter for healpy
        """
        ra = self.posterior["phi"]
        dec = self.posterior["dec"]
        distance = self.posterior["distance"]
        theta = np.pi / 2 - dec

        # mask out non physical samples;
        mask = (ra > -np.pi) * (ra < np.pi)
        mask &= (theta > 0) * (theta < np.pi)

        ra = ra[mask]
        dec = dec[mask]
        theta = theta[mask]
        distance = distance[mask]

        num_samples = len(ra)

        # calculate number of samples in each pixel
        NPIX = hp.nside2npix(nside)
        ipix = hp.ang2pix(nside, theta, ra, nest=True)
        uniq, counts = np.unique(ipix, return_counts=True)

        # create empty map and then fill in non-zero pix with density
        # estimated by fraction of total samples in each pixel
        m = np.zeros(NPIX)
        m[np.in1d(range(NPIX), uniq)] = counts
        post = m / num_samples
        post /= hp.nside2pixarea(nside)
        post /= u.sr

        # compute distance ansatz for pixels containing
        # greater than a threshold number
        min_samples_count_ansatz = 15  # FIXME: make configurable
        good_ipix = uniq[counts > min_samples_count_ansatz]
        dist_mu = []
        dist_sigma = []
        dist_norm = []
        for _ipix in good_ipix:
            _distance = distance[ipix == _ipix]
            _rho, _m, _s = dist_moments_from_samples_impl(_distance)
            _mu, _sigma, _norm = distance_ansatz_impl(_s, _m)
            dist_mu.append(_mu)
            dist_sigma.append(_sigma)
            dist_norm.append(_norm)

        mu = np.ones(NPIX) * np.inf
        mu[np.in1d(range(NPIX), good_ipix)] = np.array(dist_mu)
        mu *= u.Mpc
        sigma = np.ones(NPIX)
        sigma[np.in1d(range(NPIX), good_ipix)] = np.array(dist_sigma)
        sigma *= u.Mpc
        norm = np.zeros(NPIX)
        norm[np.in1d(range(NPIX), good_ipix)] = np.array(dist_norm)
        norm /= u.Mpc**2
        uniq_ipix = nest2uniq(nside, np.arange(NPIX))

        # convert to astropy table
        t = table.Table(
            [uniq_ipix, post, mu, sigma, norm],
            names=["UNIQ", "PROBDENSITY", "DISTMU", "DISTSIGMA", "DISTNORM"],
            copy=False,
        )
        fits_table = io.fits.table_to_hdu(t)
        # headers required for ligo.skymap cli
        fits_table.header.extend(
            [
                ("PIXTYPE", "HEALPIX", "HEALPIX pixelisation"),
                (
                    "ORDERING",
                    "NUNIQ",
                    "Pixel ordering scheme: RING, NESTED, or NUNIQ",
                ),
            ]
        )
        self.fits_table = fits_table  # attach as an attrib
        return fits_table

    def calculate_searched_area(self, nside: int):
        healpix = (
            self.get_sky_projection(nside)
            if not hasattr(self, "fits_table")
            else self.fits_table
        )
        healpix = healpix.data["PROBDENSITY"]

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
        healpix = (
            self.get_sky_projection(nside)
            if not hasattr(self, "fits_table")
            else self.fits_table
        )
        healpix = healpix.data["PROBDENSITY"]
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

    def calculate_distance_ansatz(self, nside, maxiter=10):
        """Calculate the DISTMU, DISTSIGMA, DISTNORM parameters"""
        _ = self.get_sky_projection(nside)  # set the fits_table attrib
