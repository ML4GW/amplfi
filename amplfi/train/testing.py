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
    # need double for accurate moment calculation
    dist = dist[mask].to(torch.float64)

    num_samples = len(ra)

    # calculate number of samples in each pixel
    NPIX = nside2npix(nside)

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
    post /= nside2pixarea(nside)

    # compute distance ansatz for pixels containing
    # greater than a threshold number
    good_ipix = uniq[counts > min_samples_per_pix]

    indices = torch.where(good_ipix[:, None] == ipix)
    unique_indices = torch.unique(indices[0])
    grouped_indices = []
    max_length = 0

    for idx in unique_indices:
        group = indices[1][indices[0] == idx].tolist()
        grouped_indices.append(group)
        if len(group) > max_length:
            max_length = len(group)

    _distance = torch.zeros(
        len(grouped_indices), max_length, device=device, dtype=dist.dtype
    )
    for i, _indices in enumerate(grouped_indices):
        _distance[i, : len(_indices)] = dist[_indices]

    _, _m, _s = distance.moments_from_samples_impl(_distance)
    dist_mu, dist_sigma, dist_norm = distance.ansatz_impl(_s, _m)

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

    uniq_ipix = nest2uniq(nside, NPIX_arange)

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
        true_ipix = hp.ang2pix(nside, theta_inj, ra_inj, nest=True)

        sorted_idxs = torch.argsort(healpix)[
            ::-1
        ]  # sort pixels in descending order
        # count number of pixels before hitting the pixel with injection
        # in the sorted array
        num_pix_before_injection = 1 + torch.argmax(sorted_idxs == true_ipix)
        searched_area = num_pix_before_injection * nside2pixarea(
            nside, degrees=True
        )
        healpix_cumsum = torch.cumsum(healpix[sorted_idxs])
        fifty = torch.argmin(healpix_cumsum < 0.5) * nside2pixarea(
            nside, degrees=True
        )
        ninety = torch.argmin(healpix_cumsum < 0.9) * nside2pixarea(
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
