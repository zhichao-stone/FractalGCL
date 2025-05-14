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
    def __init__(self, temperature: float = 0.4, alpha: float = 0.1, sigma: float = 0.1) -> None:
        super(FractalGCLLoss, self).__init__(temperature)
        self.alpha = alpha
        self.sigma = sigma

    def gaussian_box_dimension_random_matrix(self, 
        dimensions1: torch.Tensor, diameters1: torch.Tensor,  
        dimensions2: torch.Tensor, diameters2: torch.Tensor
    ):
        d1, d2 = np.maximum(diameters1.detach().cpu().numpy(), 2), np.maximum(diameters2.detach().cpu().numpy(), 2)
        dims1, dims2 = dimensions1.detach().cpu().numpy(), dimensions2.detach().cpu().numpy()
        tau21, tau22 = 6*(self.sigma**2) / (d1 * (np.log(d1) ** 2)), 6*(self.sigma**2) / (d2 * (np.log(d2) ** 2))
        
        mu = np.abs(dims1[:, None] - dims2[None, :])
        std = np.sqrt(tau21[:, None] + tau22[None, :])
        G_noise = np.random.normal(loc=mu, scale=std)
        return torch.from_numpy(G_noise)

    def __call__(self, 
        v1: torch.Tensor, v2: torch.Tensor, 
        dimensions1: torch.Tensor, dimensions2: torch.Tensor, 
        diameters1: torch.Tensor, diameters2: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        sim = torch.matmul(F.normalize(v1), F.normalize(v2).T) / self.temperature
        if self.alpha > 0:
            noise = self.gaussian_box_dimension_random_matrix(
                dimensions1, diameters1, 
                dimensions2, diameters2
            ).to(sim.device)
            sim = sim + self.alpha * noise

        sim = torch.exp(sim)
        pos = sim.diag()

        row_l = pos / (sim.sum(dim=1) - pos)
        col_l = pos / (sim.sum(dim=0) - pos)
        loss = - (torch.log(row_l) + torch.log(col_l)).mean() / 2

        return loss
