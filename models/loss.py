import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Loss, self).__init__(*args, **kwargs)

    def __call__(self, v1: torch.Tensor, v2: torch.Tensor, **kwargs) -> torch.Tensor:
        pass


class GCLLoss(Loss):
    def __init__(self, temperature: float = 0.5) -> None:
        super(GCLLoss, self).__init__()
        self.temperature = temperature

    def __call__(self, v1: torch.Tensor, v2: torch.Tensor, **kwargs) -> torch.Tensor:
        sim = torch.matmul(F.normalize(v1), F.normalize(v2).T) / self.temperature
        sim = torch.exp(sim)
        pos = sim.diag()

        row_l = pos / (sim.sum(dim=1) - pos)
        col_l = pos / (sim.sum(dim=0) - pos)
        loss = - (torch.log(row_l) + torch.log(col_l)).mean() / 2

        return loss


class FractalGCLLoss(GCLLoss):
    def __init__(self, temperature: float = 0.5, alpha: float = 0.01, sigma: float = 0.1) -> None:
        super(FractalGCLLoss, self).__init__(temperature)
        self.alpha = alpha
        self.sigma = sigma

    def gaussian_box_dimension_random_matrix(self, dimensions: torch.Tensor, diameters: torch.Tensor):
        tau2 = np.zeros(dimensions.size())
        for i, d in enumerate(diameters):
            D = max(d, 2)
            tau2[i] = (6*(self.sigma**2) / (D*(np.log(D)**2)))
        
        dims = dimensions.numpy()
        
        mu = np.abs(dims[:, None] - dims[None, :])
        std = np.sqrt(tau2[:, None] + tau2[None, :])
        # G_noise = np.random.normal(loc=mu, scale=std) / np.maximum.outer(dims, dims)
        G_noise = np.random.normal(loc=mu, scale=std)
        return torch.from_numpy(G_noise)

    def __call__(self, v1: torch.Tensor, v2: torch.Tensor, dimensions: torch.Tensor, diameters: torch.Tensor, **kwargs) -> torch.Tensor:
        sim = torch.matmul(F.normalize(v1), F.normalize(v2).T) / self.temperature
        if self.alpha > 0:
            noise = self.gaussian_box_dimension_random_matrix(dimensions, diameters).to(sim.device)
            sim = sim + self.alpha * noise

        sim = torch.exp(sim)
        pos = sim.diag()

        row_l = pos / (sim.sum(dim=1) - pos)
        col_l = pos / (sim.sum(dim=0) - pos)
        loss = - (torch.log(row_l) + torch.log(col_l)).mean() / 2

        return loss