from pathlib import Path

import bilby
import healpy as hp
import matplotlib.pyplot as plt
import torch
from astropy import io, table
from astropy import units as u
from ml4gw.constants import PI

from . import distance

max_nside = 1 << 29


def nest2uniq(nside, ipix):
    return 4 * nside * nside + ipix


def nside2npix(nside):
    return 12 * nside * nside


def nside2pixarea(nside, degrees=True):
    pixarea = 4 * PI / nside2npix(nside)

    if degrees:
        pixarea = pixarea * (180.0 / PI) ** 2

    return pixarea


def isnsideok(nside, nest=False):
    if hasattr(nside, "__len__"):
        if not isinstance(nside, torch.Tensor):
            nside = torch.asarray(nside)
        is_nside_ok = (
            (nside == nside.int()) & (nside > 0) & (nside <= max_nside)
        )
        if nest:
            is_nside_ok &= nside.int() & (nside.int() - 1 == 0)
    else:
        is_nside_ok = (nside == int(nside)) and (0 < nside <= max_nside)
        if nest:
            is_nside_ok = is_nside_ok and (int(nside) & (int(nside) - 1)) == 0
    return is_nside_ok


def check_nside(nside, nest=False):
    """Raises exception if nside is not valid"""
    if not torch.all(isnsideok(nside, nest=nest)):
        raise ValueError(
            f"{nside} is not a valid nside parameter (must be a power of 2,\
                less than 2**30)"
        )


def lonlat2thetaphi(lon, lat):
    return PI / 2.0 - torch.deg2rad(lat), torch.deg2rad(lon)


def check_theta_valid(theta):
    """Raises exception if theta is not within 0 and pi"""
    theta = torch.asarray(theta)
    if not ((theta >= 0).all() and (theta <= PI + 1e-5).all()):
        raise ValueError("THETA is out of range [0,pi]")


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
    theta = PI / 2 - dec
    # mask out non physical samples;
    mask = (ra > 0) * (ra < 2 * PI)
    mask &= (theta > 0) * (theta < PI)

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
        theta_inj = PI / 2 - dec_inj
        true_ipix = hp.ang2pix(nside, theta_inj, ra_inj, nest=True)

        sorted_idxs = np.argsort(healpix)[
            ::-1
        ]  # sort pixels in descending order
        # count number of pixels before hitting the pixel with injection
        # in the sorted array
        num_pix_before_injection = 1 + np.argmax(sorted_idxs == true_ipix)
        searched_area = num_pix_before_injection * hp.nside2pixarea(
            nside, degrees=True
        )
        healpix_cumsum = np.cumsum(healpix[sorted_idxs])
        fifty = np.argmin(healpix_cumsum < 0.5) * hp.nside2pixarea(
            nside, degrees=True
        )
        ninety = np.argmin(healpix_cumsum < 0.9) * hp.nside2pixarea(
            nside, degrees=True
        )
        return searched_area, fifty, ninety

    def plot_mollview(self, outpath: Path = None):
        if not hasattr(self, "fits_table"):
            raise RuntimeError("Call calculate_skymap before plotting")

        healpix = self.fits_table.data["PROBDENSITY"]
        ra_inj = self.injection_parameters["phi"]
        dec_inj = self.injection_parameters["dec"]
        theta_inj = PI / 2 - dec_inj
        plt.close()
        # plot molleweide
        plt.close()
        fig = hp.mollview(healpix, nest=True)

        # plot true values if available
        if self.injection_parameters is not None:
            ra_inj = self.injection_parameters["phi"]
            dec_inj = self.injection_parameters["dec"]
            theta_inj = np.pi / 2 - dec_inj

            hp.visufunc.projscatter(
                theta_inj, ra_inj, marker="x", color="red", s=150
            )

        plt.savefig(outpath)

        return fig

    def calculate_skymap(self, nside, min_samples_per_pix):
        """Calculate the 3D skymap. This involves the probability
        per pixel along with DISTMU, DISTSIGMA, DISTNORM parameters"""
        fits_table = get_sky_projection(
            self.posterior["phi"],
            self.posterior["dec"],
            self.posterior["distance"],
            nside=nside,
            min_samples_per_pix=min_samples_per_pix,
        )
        self.fits_table = fits_table
