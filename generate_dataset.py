import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from utils import generate_random_eigenvalues, generate_random_rotations, compute_ellipsoid_volume, \
    compute_ellipsoid_overlap

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
