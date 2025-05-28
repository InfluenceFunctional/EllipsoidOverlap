import argparse
import os
from pathlib import Path

from scipy.spatial.transform import Rotation as R
import torch
from tqdm import tqdm


def sample_triangular_right(n_samples, start, stop, device='cpu'):
    """
    sample from the CDF of a uniform distribution
    the right-aligned triangular distribution
    """
    U = torch.rand(n_samples, device=device)
    return start + (stop - start) * torch.sqrt(U)


def generate_random_eigenvalues(num_samples, device):
    """
    Generate ellipsoid eigenvector lengths according to the rule 1 <= a <= b <= c
    :return:
    """

    rands = torch.rand(num_samples, device=device).clip(min=0.5)
    b_rands = sample_triangular_right(n_samples, 0, 1.1, device).clip(min=0.1, max=1)
    c_rands = sample_triangular_right(n_samples, 0, 1.1, device).clip(min=0.1, max=1)

    a = rands
    b = a * b_rands
    c = b * c_rands

    return torch.stack([a, b, c]).T


def generate_random_rotations(num_samples, device):
    return torch.tensor(R.random(num_samples).as_matrix(), device=device, dtype=torch.float32)


def compute_ellipsoid_dot_products(e1, e2):
    """
    compute the row-to-row dot products for batches of ellipsoids [n, i, j]
    returns the [n, i, i] matrix of dot products between rows in e1 and e2

    permuting the order of e1, e2, results in transposition of the overlap matrix
    :param e1:
    :param e2:
    :return:
    """
    return torch.einsum('nij, nkj -> nik', e1, e2)


def compute_trailing_relative_diffs(record, eps: float = 1e-5):
    cum_vols = torch.cumsum(record, dim=0)
    cum_iters = torch.arange(1, len(record) + 1, device=record.device)
    cum_means = cum_vols / cum_iters
    rel_diffs = torch.diff(cum_means / (cum_means.mean() + eps))
    return rel_diffs


def compute_ellipsoid_volume(e):
    return 4 / 3 * torch.pi * e.norm(dim=-1).prod(dim=-1)


def compute_ellipsoid_overlap(e1,
                              e2,
                              v1_true,
                              v2_true,
                              r,
                              num_probes: int = 100000,
                              eps: float = 1e-3,
                              max_iters: int = 1000,
                              min_iters: int = 10,
                              show_tqdm: bool = False
                              ):
    """
    Compute the volume of the overlapping region between ellipsoids defined by e1, e2
    Done by random resampling until satisfactory convergence is achieved
    

    probe the region occupied by both ellipsoids, and check if points are within each
    :param num_probes:
    :param min_iters:
    :param eps:
    :param max_iters: 
    :param e1: [i, j] of row-wise eigenvectors defining ellipsoid 1
    :param e2: [i, j] of row-wise eigenvectors defining ellipsoid 1
    :param r: separation vector between e1 centroids and e2 centroids
    :return:
    """
    if (e1.norm(dim=1).max() + e2.norm(dim=1).max()) <= r.norm():  # impossible for them to intersect
        return 0, True

    assert len(e1) == len(e2) == len(r)

    device = e1.device
    # e1 stays on origin
    # move e2 to point r

    A1 = torch.linalg.inv(e1.T @ e1)  # Metric tensor
    A2 = torch.linalg.inv(e2.T @ e2)  # Metric tensor

    c1 = torch.zeros(3, device=device)[None, ...]
    c2 = r[None, ...]
    centers = torch.cat([c1, c2], dim=0)
    axes = torch.stack([e1, e2])  # shape: (2, 3, 3)

    # Compute corners: c ± each axis vector
    extents = axes.abs().sum(dim=1)  # shape: (2, 3), max extent in each dim
    mins = centers - extents  # shape: (2, 3)
    maxs = centers + extents

    # Bounding box that contains both ellipsoids
    bbox_min = mins.min(dim=0).values  # shape: (3,)
    bbox_max = maxs.max(dim=0).values  # shape: (3,)

    volume_record = []
    converged = False
    iter = 0

    with tqdm(total=max_iters, desc="Optimizing", unit="iter", disable=not show_tqdm) as pbar:
        while not converged and iter < max_iters:
            iter += 1

            points = bbox_min + (bbox_max - bbox_min) * torch.rand((num_probes, 3), device=device)
            bounding_volume = torch.prod(bbox_max - bbox_min)

            # check which points are inside each ellipsoid
            dx1 = points - c1
            in_e1 = (torch.einsum('ni,ij,nj->n', dx1, A1, dx1) <= 1).float()
            dx2 = points - c2
            in_e2 = (torch.einsum('ni,ij,nj->n', dx2, A2, dx2) <= 1).float()

            # Estimate ellipsoid volume
            v1 = bounding_volume * in_e1.mean()
            v2 = bounding_volume * in_e2.mean()
            v_ov = bounding_volume * (in_e1 * in_e2).mean()
            volume_record.append([float(v1), float(v2), float(v_ov)])

            if iter > min_iters:
                recs = torch.tensor(volume_record)
                v1_rec = recs[:, 0]
                v2_rec = recs[:, 1]
                ov_rec = recs[:, 2]

                # stop iterating when the mean estimates are stable
                # AND when the single volume estimates are accurate

                # stable overlap estimate
                v1_relative_diffs = compute_trailing_relative_diffs(v1_rec)
                v2_relative_diffs = compute_trailing_relative_diffs(v2_rec)
                ov_relative_diffs = compute_trailing_relative_diffs(ov_rec)

                criteria1 = v1_relative_diffs[-min(10, min_iters):].abs().mean()
                criteria2 = v2_relative_diffs[-min(10, min_iters):].abs().mean()
                criteria3 = ov_relative_diffs[-min(10, min_iters):].abs().mean()
                criteria4 = ((v1_rec.mean() - v1_true).abs() / (v1_true + 1e-5))
                criteria5 = ((v2_rec.mean() - v2_true).abs() / (v2_true + 1e-5))
                criteria = torch.tensor([criteria1, criteria2, criteria3, criteria4, criteria5])
                if torch.all(criteria < eps):  # relative change in running average less than eps
                    converged = True

                pbar.set_description(f"Iter {iter} | loss={criteria.mean():.4f}")

            pbar.update(1)

    if not converged:
        aa = 1

    return ov_rec.mean(), converged

    """
    # watch mean convergence
    
    record = ov_rec
    cum_vols = torch.cumsum(record, dim=0)
    cum_iters = torch.arange(1, len(record) + 1, device=record.device)
    cum_means = cum_vols / cum_iters
    rel_diffs = torch.diff(cum_means / (cum_means.mean() + eps))
    
    import plotly.graph_objects as go
    go.Figure(go.Scatter(y=rel_diffs.abs().cpu().detach())).show()
    """


if __name__ == '__main__':
    #Create the parser
    parser = argparse.ArgumentParser(description='Process an integer.')

    # Make the positional argument optional with nargs='?' and a default value
    parser.add_argument('chunk_ind', type=int, nargs='?', default=0,
                        help='An optional integer from the command line (default: 0)')

    args = parser.parse_args()
    chunk_ind = args.chunk_ind

    chunk_path = os.getcwd()
    device = 'cuda'
    n_samples = 10000
    num_probes = 100000
    min_iters = 5
    max_iters = 1000
    conv_eps = 1e-3

    """
    Initialize two batches of ellipsoid parameters
    1) basis vectors aligned with xyz
    2) eigenvalues (vector lengths)
    3) apply lengths to basis vectors
    4) randomly rotate ellipsoids
    
    Ellipsoids are parameterized such that the rows are the basis vectors
    with norms equal to their eigenvalues
    """
    basis_vectors_1 = torch.eye(3, device=device)[None, ...].repeat(n_samples, 1, 1)
    basis_vectors_2 = torch.eye(3, device=device)[None, ...].repeat(n_samples, 1, 1)

    eigvals_1 = generate_random_eigenvalues(n_samples, device)
    eigvals_2 = generate_random_eigenvalues(n_samples, device)

    rot_1 = generate_random_rotations(n_samples, device)
    rot_2 = generate_random_rotations(n_samples, device)

    ell_1 = torch.einsum('nij,nj->nij', basis_vectors_1, eigvals_1)
    ell_2 = torch.einsum('nij,nj->nij', basis_vectors_2, eigvals_2)

    ell_1 = torch.einsum('nij, nkj->nki', rot_1, ell_1)
    ell_2 = torch.einsum('nij, nkj->nki', rot_2, ell_2)

    """
    Initialize separation vectors
    1) sample separation vector length
    2) sample separation vector direction
    
    separation vector maximum length is the sum of the two largest eigenvalues
    minimum length is 0
    """
    # vector length
    rlens = (torch.rand(n_samples, device=device) * (
            eigvals_1[:, 0] + eigvals_2[:, 0])) * 0.99  # enforce at least 1% overlap
    # unit direction vector
    rdirs = torch.randn((n_samples, 3), device=device)
    rdirs = rdirs / rdirs.norm(dim=1, keepdim=True)

    rvecs = rlens[:, None] * rdirs

    """
    Get system properties
    1) Compute ellipsoid volumes
    2) Compute overlaps    
    """
    v_1 = compute_ellipsoid_volume(ell_1)
    v_2 = compute_ellipsoid_volume(ell_2)

    overlaps = torch.zeros(n_samples, device=device, dtype=torch.float32)
    converged = torch.empty(n_samples, device=device, dtype=torch.bool)
    for ind in tqdm(range(n_samples)):
        overlaps[ind], converged[ind] = compute_ellipsoid_overlap(
            ell_1[ind],
            ell_2[ind],
            v_1[ind],
            v_2[ind],
            rvecs[ind],
            show_tqdm=False,
            num_probes=num_probes,
            min_iters=min_iters,
            max_iters=max_iters,
            eps=conv_eps)

    results_dict = {
        'overlaps': overlaps[converged],
        'vol_1': v_1[converged],
        'vol_2': v_2[converged],
        'e_1': ell_1[converged],
        'e_2': ell_2[converged],
        'r': rvecs[converged],
    }

    file_path = Path(chunk_path).joinpath(Path(f'ellipsoid_data_chunk{chunk_ind}.pt'))
    torch.save(results_dict, file_path)
    end = True

"""
# show results

# raw overlap volume
import plotly.graph_objects as go
go.Figure(go.Histogram(x=overlaps.cpu().detach(), nbinsx=50)).show()

# normed overlap volume
import plotly.graph_objects as go  
go.Figure(go.Histogram(x=(overlaps.cpu().detach()) / (v_1*v_2/(v_1+v_2)).cpu().detach(), nbinsx=50)).show()

"""
