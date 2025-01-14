from pathlib import Path

import bilby
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy import io, table
from astropy import units as u

from . import distance


def nest2uniq(nside, ipix):
    return 4 * nside * nside + ipix


def get_sky_projection(ra, dec, dist, nside=32, min_samples_per_pix=15):
    """Return 3D sky localization from ra, dec,
    distance samples

    Args:
        ra: right ascension samples
        dec: declination samples
        dist: distance samples
        nside: nside parameter for HEALPix
        min_samples_per_pix: minimum # samples per pixel for distance ansatz
    """
    theta = np.pi / 2 - dec
    # mask out non physical samples;
    mask = (ra > -np.pi) * (ra < np.pi)
    mask &= (theta > 0) * (theta < np.pi)

    ra = ra[mask]
    dec = dec[mask]
    theta = theta[mask]
    dist = dist[mask]

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
    good_ipix = uniq[counts > min_samples_per_pix]
    dist_mu = []
    dist_sigma = []
    dist_norm = []
    for _ipix in good_ipix:
        _distance = dist[ipix == _ipix]
        _, _m, _s = distance.moments_from_samples_impl(_distance)
        _mu, _sigma, _norm = distance.ansatz_impl(_s, _m)
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
    return fits_table


class Result(bilby.result.Result):
    def calculate_searched_area(self, nside: int):
        if not hasattr(self, "fits_table"):
            raise RuntimeError("Call calculate_skymap before searched area")
        healpix = self.fits_table.data["PROBDENSITY"]

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

    def plot_mollview(self, outpath: Path = None):
        if not hasattr(self, "fits_table"):
            raise RuntimeError("Call calculate_skymap before plotting")
        healpix = self.fits_table.data["PROBDENSITY"]
        ra_inj = self.injection_parameters["phi"]
        dec_inj = self.injection_parameters["dec"]
        theta_inj = np.pi / 2 - dec_inj
        plt.close()
        # plot molleweide
        fig = hp.mollview(healpix, nest=True)
        hp.visufunc.projscatter(
            theta_inj, ra_inj, marker="x", color="red", s=150
        )

        plt.savefig(outpath)

        return fig

    def calculate_skymap(self, nside, min_samples_per_pix):
        """Calculate the 3D skymap. This involves the probability
        per pixel along with DISTMU, DISTSIGMA, DISTNORM parameters"""
        fits_table = get_sky_projection(
            self.posterior["ra"],
            self.posterior["dec"],
            self.posterior["distance"],
            nside=nside,
            min_samples_per_pix=min_samples_per_pix,
        )
        self.fits_table = fits_table
