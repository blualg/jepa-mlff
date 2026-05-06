from __future__ import annotations

import math

import torch
from torch import nn


class GaussianRBF(nn.Module):
    def __init__(self, n_rbf: int = 32, cutoff: float = 5.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.gamma = nn.Parameter(torch.tensor(10.0 / cutoff), requires_grad=False)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distances.unsqueeze(-1) - self.centers) ** 2)


class SchNetBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_rbf: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, rbf: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        filters = self.filter_net(rbf)
        messages = h.unsqueeze(1) * filters
        messages = messages * neighbor_mask.unsqueeze(-1)
        agg = messages.sum(dim=2)
        return self.norm(h + self.update_net(agg))


class AtomisticEncoder(nn.Module):
    def __init__(
        self,
        max_z: int = 100,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_rbf: int = 32,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.embedding = nn.Embedding(max_z + 1, hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.mask_token, std=1.0 / math.sqrt(hidden_dim))
        self.rbf = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
        self.blocks = nn.ModuleList([SchNetBlock(hidden_dim, n_rbf) for _ in range(n_layers)])

    def geometry(self, R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        diff = R.unsqueeze(2) - R.unsqueeze(1)
        distances = torch.linalg.norm(diff + 1e-9, dim=-1)
        n_atoms = R.shape[1]
        eye = torch.eye(n_atoms, dtype=torch.bool, device=R.device).unsqueeze(0)
        neighbor_mask = (distances < self.cutoff) & (~eye)
        return distances, neighbor_mask.float()

    def forward(
        self,
        z: torch.Tensor,
        R: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.embedding(z)
        if atom_mask is not None:
            h = torch.where(atom_mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), h)
        distances, neighbor_mask = self.geometry(R)
        rbf = self.rbf(distances)
        for block in self.blocks:
            h = block(h, rbf, neighbor_mask)
        return h


class JEPAPredictor(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ForceField(nn.Module):
    def __init__(self, encoder: AtomisticEncoder):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("energy_offset", torch.zeros(()))
        h = encoder.hidden_dim
        self.energy_head = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )

    def energy(self, z: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        h = self.encoder(z, R)
        atomic_e = self.energy_head(h).squeeze(-1)
        return atomic_e.sum(dim=1) + self.energy_offset

    def energy_and_forces(self, z: torch.Tensor, R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        R_req = R.detach().clone().requires_grad_(True)
        E = self.energy(z, R_req)
        grad = torch.autograd.grad(E.sum(), R_req, create_graph=self.training)[0]
        return E, -grad
