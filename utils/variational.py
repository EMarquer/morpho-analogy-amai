
import torch
from torch import nn
from torch.nn import functional as F
from typing import *

class VarLayer(nn.Module):
    # https://github.com/williamFalcon/pytorch-lightning-vae/blob/main/vae.py
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 **kwargs) -> None:
        super(VarLayer, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # distribution parameters
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_var = nn.Linear(input_dim, latent_dim)
        #self.beta = beta

        # for the gaussian likelihood
        #self.log_scale = nn.Parameter(torch.Tensor([0.0]))


    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl


    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(input)
        log_var = self.fc_var(input)

        return [mu, log_var]


    def reparameterize(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * var)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z, std

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        mu, var = self.encode(input)
        z, std = self.reparameterize(mu, var)

        kl = self.kl_divergence(z, mu, std)

        return  z, (mu, var), kl

    def loss_function(self,
                      mu: torch.Tensor, log_var: torch.Tensor,
                      kld_weight=1.0, **kwargs) -> dict:

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        return kld_loss, (self.beta * kld_weight * kld_loss).mean()

    # def sample(self,
    #            num_samples:int,
    #            device, **kwargs) -> torch.Tensor:
    #     """
    #     Samples from the latent space.
    #     :param num_samples: (Int) Number of samples
    #     :param device: Device to run the model
    #     :return: (Tensor) [num_samples, latent_dim]
    #     """
    #     torch.distributions.Normal(mu, std)
    #     z = torch.randn(num_samples,
    #                     self.latent_dim,
    #                     device=device)

    #     return z













# class BetaVAEH(nn.Module):
#     def __init__(self,
#                  input_dim: int,
#                  latent_dim: int,
#                  beta: int = 4,
#                  **kwargs) -> None:
#         super(BetaVAEH, self).__init__()

#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
#         self.fc_mu = nn.Linear(input_dim, latent_dim)
#         self.fc_var = nn.Linear(input_dim, latent_dim)
#         self.beta = beta

#     def encode(self, result: torch.Tensor) -> List[torch.Tensor]:
#         """
#         Encodes the input by passing through the encoder network
#         and returns the latent codes.
#         :param input: (Tensor) Input tensor to encoder [N x C x H x W]
#         :return: (Tensor) List of latent codes
#         """
#         # Split the result into mu and var components
#         # of the latent Gaussian distribution
#         mu = self.fc_mu(result)
#         log_var = self.fc_var(result)

#         return [mu, log_var]

#     def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         """
#         Will a single z be enough ti compute the expectation
#         for the loss??
#         :param mu: (Tensor) Mean of the latent Gaussian
#         :param logvar: (Tensor) Standard deviation of the latent Gaussian
#         :return:
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu

#     def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
#         mu, log_var = self.encode(input)
#         z = self.reparameterize(mu, log_var)
#         return  z, mu, log_var

#     def loss_function(self,
#                       mu: torch.Tensor, log_var: torch.Tensor,
#                       kld_weight=1.0, **kwargs) -> dict:

#         kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

#         return kld_loss, (self.beta * kld_weight * kld_loss).mean()

#     def sample(self,
#                num_samples:int,
#                device, **kwargs) -> torch.Tensor:
#         """
#         Samples from the latent space.
#         :param num_samples: (Int) Number of samples
#         :param device: Device to run the model
#         :return: (Tensor) [num_samples, latent_dim]
#         """
#         z = torch.randn(num_samples,
#                         self.latent_dim,
#                         device=device)

#         return z

# class BetaVAEB(BetaVAEH):
#     def __init__(self,
#                  input_dim: int,
#                  latent_dim: int,
#                  beta: int = 4,
#                  gamma:float = 1000.,
#                  max_capacity: int = 25,
#                  Capacity_max_iter: int = 1e5,
#                  **kwargs) -> None:
#         super(BetaVAEB, self).__init__(input_dim, latent_dim, beta=beta)

#         self.num_iter = 0
#         self.gamma = gamma
#         self.C_max = torch.Tensor([max_capacity])
#         self.C_stop_iter = Capacity_max_iter

#     def loss_function(self,
#                         mu: torch.Tensor, log_var: torch.Tensor, batch_idx,
#                       kld_weight=1.0,
#                       **kwargs) -> dict:
#         self.num_iter += 1

#         kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

#         # https://arxiv.org/pdf/1804.03599.pdf
#         self.C_max = self.C_max.to(mu.device)
#         C = torch.clamp(self.C_max/self.C_stop_iter * batch_idx, 0, self.C_max.data[0])
#         loss = self.gamma * kld_weight * (kld_loss - C).abs()
        
#         return kld_loss.mean(), loss.mean()

# #BetaVAE = BetaVAEB

# #def weighted_cosine
# #def weighted_euclidean

# def variational_closest(mu: torch.Tensor, log_var: torch.Tensor, tensors: torch.Tensor):
#     """Find the closest vector, after normalizing the space by mean and the inverse of standard deviation.

#     Can be interpreted as a norm weighted by the inverse of standerd deviation:
#     the larger the variance, the lower the importance of the dimension in the sum.

#     :param mu: mean vector [latent_dim]
#     :param log_var: log(var) vector [latent_dim]
#     :param tensors: batch of reference vectors to explore [b, latent_dim]
#     :return: normalized distances, indices in increasing order of distances, closest
#     """
#     std = log_var.exp().sqrt()
#     normalized = tensors - mu.expand(tensors.size()) / std.expand(tensors.size())
#     distances = normalized.norm(dim=-1)
#     ranked_closest = distances.argsort()
#     return distances, ranked_closest, ranked_closest[0]

#     # for a cosine variant see https://stats.stackexchange.com/questions/384419/weighted-cosine-similarity