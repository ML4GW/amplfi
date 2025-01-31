from pathlib import Path

import bilby
import healpy as hp
import matplotlib.pyplot as plt
import ml4gw.utils.healpix as mlhp
import torch
from astropy import io, table
from astropy import units as u
from ml4gw.constants import PI

from . import distance


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
    mask = (ra > -PI) * (ra < PI)
    mask &= (theta > 0) * (theta < PI)

    ra = ra[mask]
    dec = dec[mask]
    theta = theta[mask]
    dist = dist[mask]

    num_samples = len(ra)

    # calculate number of samples in each pixel
    NPIX = mlhp.nside2npix(nside)

    device = theta.device
    NPIX_arange = torch.arange(NPIX, device=device)

    theta, ra = theta.to("cpu"), ra.to("cpu")
    ipix = hp.ang2pix(nside, theta, ra, nest=True)
    theta, ra, ipix = theta.to(device), ra.to(device), ipix.to(device)
    uniq, counts = torch.unique(ipix, return_counts=True)

    # create empty map and then fill in non-zero pix with density
    # estimated by fraction of total samples in each pixel
    m = torch.zeros(NPIX, device=device)
    m[torch.isin(NPIX_arange, uniq)] = counts.to(m.dtype)
    post = m / num_samples
    post /= mlhp.nside2pixarea(nside)

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

    mu = torch.ones(NPIX, device=device) * torch.inf
    mu[torch.isin(NPIX_arange, good_ipix)] = torch.tensor(
        dist_mu, device=device, dtype=mu.dtype
    )

    sigma = torch.ones(NPIX, device=device)
    sigma[torch.isin(NPIX_arange, good_ipix)] = torch.tensor(
        dist_sigma, device=device, dtype=sigma.dtype
    )

    norm = torch.zeros(NPIX, device=device)
    norm[torch.isin(NPIX_arange, good_ipix)] = torch.tensor(
        dist_norm, device=device, dtype=norm.dtype
    )

    uniq_ipix = mlhp.nest2uniq(nside, NPIX_arange)

    uniq_ipix = uniq_ipix.cpu().numpy()
    post = post.cpu().numpy() * u.sr
    mu = mu.cpu().numpy() * u.Mpc
    sigma = sigma.cpu().numpy() * u.Mpc
    norm = norm.cpu().numpy() / u.Mpc / u.Mpc

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
        true_ipix = hp.ang2pix(nside, theta_inj, ra_inj)

        sorted_idxs = torch.argsort(healpix)[
            ::-1
        ]  # sort pixels in descending order
        # count number of pixels before hitting the pixel with injection
        # in the sorted array
        num_pix_before_injection = 1 + torch.argmax(sorted_idxs == true_ipix)
        searched_area = num_pix_before_injection * mlhp.nside2pixarea(
            nside, degrees=True
        )
        return searched_area

    def plot_mollview(self, outpath: Path = None):
        if not hasattr(self, "fits_table"):
            raise RuntimeError("Call calculate_skymap before plotting")
        healpix = self.fits_table.data["PROBDENSITY"]
        ra_inj = self.injection_parameters["phi"]
        dec_inj = self.injection_parameters["dec"]
        theta_inj = PI / 2 - dec_inj
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
            self.posterior["phi"],
            self.posterior["dec"],
            self.posterior["distance"],
            nside=nside,
            min_samples_per_pix=min_samples_per_pix,
        )
        self.fits_table = fits_table
