import plotly.graph_objects as go
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from utils import compute_ellipsoid_dot_products


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 norm, dropout_p):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        if norm is not None:
            self.norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
        else:
            self.norms = nn.ModuleList([
                nn.Identity() for _ in range(num_layers)
            ])

        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_p) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer, dropout, norm in zip(self.hidden_layers, self.dropouts, self.norms):
            residual = x
            x = dropout(F.relu(norm(layer(x))))
            x = x + residual  # skip connection
        return self.output_layer(x)


class EllipsoidsDataset(Dataset):
    def __init__(self, e1, e2, r, v1, v2, ov):
        """
        Below parameterization is not permutation invariant, so we do
        duplicative sample augmentation
        """
        self.e1 = torch.tensor(torch.cat([e1, e2]), dtype=torch.float32)
        self.e2 = torch.tensor(torch.cat([e2, e1]), dtype=torch.float32)
        self.r = torch.tensor(torch.cat([r, r]), dtype=torch.float32)
        self.v1 = torch.tensor(torch.cat([v1, v2]), dtype=torch.float32)
        self.v2 = torch.tensor(torch.cat([v2, v1]), dtype=torch.float32)
        self.ov = torch.tensor(torch.cat([ov, ov]), dtype=torch.float32)

        self.normalize_dataset()
        self.standardize_directions()
        self.parameterize_dataset()

    def parameterize_dataset(self):
        """
            parameterization is
            0: length of r
            1-9: dot product between e1 and e2,
            10-15: eigenvalues
            """
        r_hat = F.normalize(self.r, dim=-1)
        r1_local = torch.einsum('nij,nj->ni', self.e1, r_hat)  # r in frame of ellipsoid 1
        r2_local = torch.einsum('nij,nj->ni', self.e2, -r_hat)  # r in frame of ellipsoid 2

        normed_e1 = self.e1 / self.e1.norm(dim=-1, keepdim=True)
        normed_e2 = self.e2 / self.e2.norm(dim=-1, keepdim=True)

        R_rel = torch.einsum('nik, njk -> nij', normed_e1, normed_e2)

        self.x = torch.zeros((len(self.e1), 31), dtype=torch.float32)
        self.x[:, 0] = self.r.norm(dim=-1)
        self.x[:, 1:10] = compute_ellipsoid_dot_products(normed_e1, normed_e2).reshape(len(self.e1), 9)
        self.x[:, 10:13] = self.e1.norm(dim=-1)
        self.x[:, 13:16] = self.e2.norm(dim=-1)
        self.x[:, 16:19] = r1_local
        self.x[:, 19:22] = r2_local
        self.x[:, 22:31] = R_rel.reshape(len(self.e1), 9)

        # self.x = torch.cat([
        #     self.r.norm(dim=-1)[:, None],
        #     self.e1.flatten(1, 2),
        #     self.e2.flatten(1, 2),
        # ], dim=1)
        # self.x = torch.cat([
        #     self.r,
        #     self.e1.flatten(1, 2),
        #     self.e2.flatten(1, 2),
        # ], dim=1)

        ov_target = self.ov / (self.v1 * self.v2 / (self.v1 + self.v2))
        self.y = torch.stack([self.v1, self.v2, ov_target]).T

    def standardize_directions(self):
        """
            standardize e1 and e2 directions by forcing them to point at one another
            e1 is always at the origin so it just points towards r
            e2 is at r, pointing along -r to the origin
            """
        dot1 = torch.einsum('nij,ni->nj', self.e2, self.r)
        sign_flip1 = (dot1 > 0).float() * -2 + 1  # flips if dot2 > 0, i.e., points same way as r
        self.e1 = self.e1 * sign_flip1.unsqueeze(-1)
        dot2 = torch.einsum('nij,ni->nj', self.e2, self.r)
        sign_flip2 = (dot2 > 0).float() * -2 + 1  # flips if dot2 > 0, i.e., points same way as r
        self.e2 = self.e2 * sign_flip2.unsqueeze(-1)

    def normalize_dataset(self):
        """
            since relative overlap is scale-invariant,
            we'll normalize all values by the largest eigenvalue in each sample
            """
        max_e1 = self.e1.norm(dim=-1).amax(1)
        max_e2 = self.e2.norm(dim=-1).amax(1)
        max_val = torch.stack([max_e1, max_e2]).T.amax(1)

        self.e1 /= max_val[:, None, None]
        self.e2 /= max_val[:, None, None]
        self.r /= max_val[:, None]
        self.v1 /= max_val ** 3
        self.v2 /= max_val ** 3
        self.ov /= max_val ** 3

    def __len__(self):
        return len(self.r)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def parity_plot(target, output, loss):
    targets = target[:, 2].cpu().detach()
    predictions = output[:, 2].cpu().detach()
    errors = loss[:, 2].cpu().detach()
    corrcoef = torch.corrcoef(torch.stack([targets, predictions]))[0, 1]

    x_min = min([targets.amin(), predictions.amin()])
    x_max = max([targets.amax(), predictions.amax()])
    fig = go.Figure()
    fig.add_scatter(x=targets,
                    y=predictions,
                    marker_color=torch.log(1 + errors),
                    name=f"R={corrcoef:.4f}",
                    showlegend=True,
                    opacity=.9,
                    mode='markers',
                    marker_colorscale='bluered')

    fig.update_layout(xaxis_range=[x_min, x_max],
                      yaxis_range=[x_min, x_max],
                      xaxis_title='Target Overlap',
                      yaxis_title='Predicted Overlap')

    return fig, corrcoef


def checkpointing(model, test_loss):
    past_mean_losses = torch.tensor([torch.mean(record[:, 2]) for record in test_loss])
    current_loss = torch.mean(test_loss[-1][:, 2])
    if current_loss == torch.amin(past_mean_losses):
        torch.save(model, 'best_overlap_model.pt')


def run_epoch(model, optimizer, tr, te, device, scheduler, epoch):
    model.train()
    train_loss = []
    for batch_idx, (data, target) in enumerate(tr):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.smooth_l1_loss(output, target, reduction='none')
        if loss[:, :2].mean() < 5e-3:  # exclude volume training if it's converged
            loss[:, 2].mean().backward()
        else:
            loss.mean().backward()
        optimizer.step()
        train_loss.append(loss.cpu().detach())

    model.eval()
    test_loss = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(te):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.smooth_l1_loss(output, target, reduction='none')
            test_loss.append(loss.cpu().detach())
    scheduler.step(epoch)

    return train_loss, test_loss, target, output, loss

def get_dataloaders(batch_size, train_dataset_path, test_dataset_path):

    train_dataset = torch.load(train_dataset_path)

    tr = DataLoader(EllipsoidsDataset(
        train_dataset['e_1'],
        train_dataset['e_2'],
        train_dataset['r'],
        train_dataset['vol_1'],
        train_dataset['vol_2'],
        train_dataset['overlaps']
    ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)

    test_dataset = torch.load(test_dataset_path)

    te = DataLoader(EllipsoidsDataset(
        test_dataset['e_1'],
        test_dataset['e_2'],
        test_dataset['r'],
        test_dataset['vol_1'],
        test_dataset['vol_2'],
        test_dataset['overlaps']
    ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)

    return tr, te