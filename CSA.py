from typing import Optional, Callable
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from gcn import GCN
from torch_geometric.nn import GCNConv
class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: Callable,
                 base_model = GCNConv,
                 k: int = 2):

        super(Encoder, self).__init__()
        self.base_model = base_model
        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
        for _ in range(1, k - 1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)
        self.activation = activation

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class CSA(torch.nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 n_enc_out: int,
                 n_rec: int,
                 att_hid: int,
                 device: torch.device,
                 num_proj_hidden: int,
                 tau: float = 0.5,
                 ):
        super(CSA, self).__init__()
        self.encoder = encoder
        self.tau = tau
        self.fc1 = torch.nn.Linear(n_enc_out, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, n_enc_out)
        self.att = torch.nn.Sequential(
            nn.Linear(n_enc_out, att_hid),
            nn.Tanh(),
            nn.Linear(att_hid, 1, bias=False)
        )

        self.decoder = GCN(n_enc_out, n_rec, act='prelu')
        self.device = device

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self,
                   z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def _sim(self,
             z1: torch.Tensor,
             z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def _team_up(self,
                 z1: torch.Tensor,
                 z2: torch.Tensor,
                 cs: torch.Tensor,
                 current_ep: int,
                 t0: int,
                 gamma_max: int, ) -> torch.Tensor:
        gamma = min(max(0, (current_ep - t0) / 100), gamma_max)
        temp = lambda x: torch.exp(x / self.tau)
        refl_sim = temp(self._sim(z1, z1) + gamma * cs + gamma * cs.unsqueeze(dim=1))
        between_sim = temp(self._sim(z1, z2) + gamma * cs + gamma * cs.unsqueeze(dim=1))
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        return loss

    def team_up_loss(self,
                     z1: torch.Tensor,
                     z2: torch.Tensor,
                     cs: np.ndarray,
                     current_ep: int,
                     t0: int = 0,
                     gamma_max: int = 1,
                     mean: bool = True,
                     ) -> torch.Tensor:
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        cs = torch.from_numpy(cs).to(h1.device)
        l1 = self._team_up(h1, h2, cs, current_ep, t0, gamma_max)
        l2 = self._team_up(h2, h1, cs, current_ep, t0, gamma_max)
        ret = 0.5 * (l1 + l2)
        ret = ret.mean() if mean else ret.sum()
        return ret

    def z_att(self,
              z_1: torch.Tensor,
              z_2: torch.Tensor=None,

              ):
        if z_2 is None:
            z = z_1
        else:
            z = torch.stack([z_1, z_2], dim=1)
            w = self.att(z)
            alpha = torch.softmax(w, dim=1)
            z = alpha * z
            z = (z[:, 0, :] + z[:, 1, :]) / 2
        return z, alpha.squeeze()  # (n,3)

    def reconstr_loss(self,
                      z_1: torch.Tensor, # default spatial
                      weighted_adj,
                      features: torch.Tensor,
                      z_2: Optional[torch.Tensor] = None,
                      # default histology
                      ):
        # assert z_1.shape[0] == z_2.shape[0]
        if not z_2:
            z_rec = self.decoder(z_1, weighted_adj)

        else:
            z = torch.stack([z_1, z_2], dim=1)
            w = self.att(z)
            alpha = torch.softmax(w, dim=1)
            z = alpha * z
            z_avg = (z[:, 0, :] + z[:, 1, :]) / 2
            z_rec = self.decoder(z_avg, weighted_adj)

        recon_loss = torch.mean((z_rec - features) ** 2)

        return recon_loss, z_rec

