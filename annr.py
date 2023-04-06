import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from typing import Literal

class ANNr(nn.Module):
    def __init__(self, emb_size, mode = "ab!=ac", **kwargs):
        """Model:
        - d = f3(f1(a, b), f2(a, c))
        
        :param mode: if equal to "ab=ac", f1 will be used in place of f2: 
            d = f3(f1(a, b), f1(a, c))
        """
        super().__init__()
        self.emb_size = emb_size
        self.ab = nn.Linear(2 * self.emb_size, 2 * self.emb_size)
        if mode == "ab=ac":
            self.ac = self.ab
        else:
            self.ac = nn.Linear(2 * self.emb_size, 2 * self.emb_size)
        self.d = nn.Linear(4 * self.emb_size, self.emb_size)

    def forward(self, a, b, c, p=0):

        if p>0:
            a=torch.nn.functional.dropout(a, p)
            b=torch.nn.functional.dropout(b, p)
            c=torch.nn.functional.dropout(c, p)

        ab = self.ab(torch.cat([a, b], dim = -1))
        ac = self.ac(torch.cat([a, c], dim = -1))

        d = self.d(torch.cat([ab, ac], dim = -1))
        return d

class AnalogyRegressionLoss(nn.Module):
    
    def __init__(self, variant: Literal["mse", "cosine embedding loss", "relative shuffle", "relative all", "all"]="relative shuffle"):
        nn.Module.__init__(self)
        
        self.variant = variant
        assert self.variant in {"mse", "cosine embedding loss", "relative shuffle", "relative all", "all"}
        if self.variant == "cosine embedding loss" or self.variant == "all":
            self.cosine_embedding_loss = nn.CosineEmbeddingLoss()

    def forward(self, a, b, c, d, d_pred):
        if self.variant == "mse":
            return mse_loss(d_pred,d)
        elif self.variant == "cosine embedding loss":
            return self.cosine_embedding_loss(
                    torch.cat([d_pred]*4, dim=0),
                    torch.cat([d,a,b,c], dim=0),
                    torch.cat([torch.ones(a.size(0)), -torch.ones(a.size(0) * 3)]).to(a.device))

        elif self.variant == "relative shuffle":
            good = mse_loss(d, d_pred)
            bad = mse_loss(d[torch.randperm(d.size(0))], d_pred)

            return (good + 1) / (bad + 1)

        elif self.variant == "relative all":
            return (1 + mse_loss(d_pred, d) * 6) / (1 +
                mse_loss(a,b) +
                mse_loss(a,c) +
                mse_loss(a,d) +
                mse_loss(b,c) +
                mse_loss(b,d) +
                mse_loss(c,d))
        
        else:
            good = mse_loss(d, d_pred)
            bad = mse_loss(d[torch.randperm(d.size(0))], d_pred)
            return (
                self.cosine_embedding_loss(
                    torch.cat([d_pred]*4, dim=0),
                    torch.cat([d,a,b,c], dim=0),
                    torch.cat([torch.ones(a.size(0)), -torch.ones(a.size(0) * 3)]).to(a.device)) 
                    + (good + 1) / (bad + 1)
                    + ((1 + mse_loss(d_pred, d) * 6) / (1 +
                mse_loss(a,b) +
                mse_loss(a,c) +
                mse_loss(a,d) +
                mse_loss(b,c) +
                mse_loss(b,d) +
                mse_loss(c,d))))