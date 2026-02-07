import torch
from ml4gw.gw import get_ifo_geometry, compute_antenna_responses


def sample_rescaled_distance(parameters):
    """
    Rescale Distance only with Chirp Mass
    """
    M_ref = 1.0
    chirp_mass = parameters["chirp_mass"]
    distance = parameters["distance"]
    return (chirp_mass / M_ref) ** (5 / 6) * distance


def sample_chirp_distance(parameters):
    """
    Rescale Distance only with Chirp Mass
    """
    ifos = ["H1", "L1", "V1"]  ## TODO: change to config inputs
    detector_tensors, _ = get_ifo_geometry(*ifos)

    dec = parameters["dec"]
    phi = parameters["phi"]
    psi = parameters["psi"]
    inclination = parameters["inclination"]
    distance = parameters["distance"]

    theta = torch.pi / 2 - dec
    antenna_responses = compute_antenna_responses(
        theta, psi, phi, detector_tensors, ["plus", "cross"]
    )
    f_plus, f_cross = antenna_responses[:, 0], antenna_responses[:, 1]

    d_effs = []

    for i in range(len(ifos)):
        factor = (
            f_plus[:, i] * (1 + torch.cos(inclination) ** 2)
        ) ** 2 / 4 + (f_cross[:, i] * torch.cos(inclination) ** 2) ** 2
        d_eff = distance * torch.sqrt(factor)
        d_effs.append(d_eff)

    distances = torch.vstack(d_effs)
    distances, _ = torch.max(distances, dim=0)
    parameters["distance"] = distances
    return sample_rescaled_distance(parameters)
