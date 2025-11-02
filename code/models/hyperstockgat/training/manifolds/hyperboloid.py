"""Hyperboloid manifold."""

import torch

from models.hyperstockgat.training.manifolds.base import Manifold
from models.hyperstockgat.training.utils.math_utils import arcosh, cosh, sinh 


class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class.

    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K

    c = 1 / K is the hyperbolic curvature. 
    """

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def sqdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        # clamp distance to avoid nans in Fermi-Dirac decoder
        return torch.clamp(sqdist, max=50.0)

    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        # Handle both batched and non-batched inputs
        if x.dim() == 3:
            # Batched: [B, N, d+1]
            y = x[:, :, 1:]  # [B, N, d]
            y_sqnorm = torch.norm(y, p=2, dim=-1, keepdim=True) ** 2  # [B, N, 1]
            mask = torch.ones_like(x)
            mask[:, :, 0] = 0
            vals = torch.zeros_like(x)
            vals[:, :, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
            return vals + mask * x

        else:
            # Non-batched: [N, d+1]
            y = x[:, 1:]  # [N, d]
            y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
            mask = torch.ones_like(x)
            mask[:, 0] = 0
            vals = torch.zeros_like(x)
            vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
            return vals + mask * x


    def proj_tan(self, u, x, c):
        K = 1. / c
        d = x.size(-1) - 1  # last dimension is features
        # handle both batched [B, N, D] and unbatched [N, D]
        if u.dim() == 3:
            ux = torch.sum(x[:, :, 1:] * u[:, :, 1:], dim=-1, keepdim=True)
            mask = torch.ones_like(u)
            mask[:, :, 0] = 0
            vals = torch.zeros_like(u)
            vals[:, :, 0:1] = ux / torch.clamp(x[:, :, 0:1], min=self.eps[x.dtype])
            return vals + mask * u
        else:
            ux = torch.sum(x[:, 1:] * u[:, 1:], dim=1, keepdim=True)
            mask = torch.ones_like(u)
            mask[:, 0] = 0
            vals = torch.zeros_like(u)
            vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
            return vals + mask * u

    def proj_tan0(self, u, c):
        if u.dim() == 3:
            vals = torch.zeros_like(u)
            vals[:, :, 0:1] = u[:, :, 0:1]
            return u - vals
        else:
            vals = torch.zeros_like(u)
            vals[:, 0:1] = u[:, 0:1]
            return u - vals

    def expmap(self, u, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)

        if u.dim() == 3:
            # Batched: [B, N, D]
            result = cosh(theta) * x + sinh(theta) * u / theta
            return self.proj(result, c)
        else:
            # Unbatched: [N, D]
            result = cosh(theta) * x + sinh(theta) * u / theta
            return self.proj(result, c)
        
    def logmap(self, x, y, c):
        K = 1. / c
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        sinh, cosh = torch.sinh, torch.cosh
        # Handle both [N, d+1] and [B, N, d+1]
        if u.dim() == 3:
            # Batched input
            B, N, D = u.size()
            d = D - 1  # feature dimension minus time-like coord
            x = u[:, :, 1:]                     # [B, N, d]
            x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)  # [B, N, 1]
            x_norm = torch.clamp(x_norm, min=self.min_norm)
            theta = x_norm / sqrtK              # [B, N, 1]

            res = torch.ones_like(u)            # [B, N, D]
            theta = torch.clamp(theta, max=15.0)  # or max=10.0

            res[:, :, 0:1] = sqrtK * cosh(theta)
            res[:, :, 1:] = sqrtK * sinh(theta) * x / x_norm
            return self.proj(res, c)

        else:
            # Non-batched input
            N, D = u.size()
            d = D - 1
            x = u[:, 1:]
            x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
            x_norm = torch.clamp(x_norm, min=self.min_norm)
            theta = x_norm / sqrtK
            theta = torch.clamp(theta, max=15.0)  # or max=10.0

            res = torch.ones_like(u)
            res[:, 0:1] = sqrtK * cosh(theta)
            res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
            return self.proj(res, c)
        
    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1

        if x.dim() == 3:
            # Batched: [B, N, d+1]
            y = x[:, :, 1:]  # spatial part
            y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
            y_norm = torch.clamp(y_norm, min=self.min_norm)

            theta = torch.clamp(x[:, :, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
            res = torch.zeros_like(x)
            res[:, :, 1:] = sqrtK * arcosh(theta) * y / y_norm
            return res

        else:
            # Non-batched: [N, d+1]
            y = x[:, 1:]
            y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
            y_norm = torch.clamp(y_norm, min=self.min_norm)

            theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
            res = torch.zeros_like(x)
            res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
            return res


    def mobius_add(self, x, y, c):
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_matvec(self, m, x, c):
        u = self.logmap0(x, c)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu, c)

    def ptransp(self, x, y, u, c):
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def ptransp0(self, x, u, c):
        """
        Parallel transport from the origin to point x on the hyperboloid.
        Handles both [N, D] and [B, N, D] shaped tensors.
        """
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1

        if x.dim() == 3:  # [B, N, D]
            B, N, D = x.size()
            x0 = x[:, :, 0:1]
            y = x[:, :, 1:]
            y_norm = torch.clamp(torch.norm(y, p=2, dim=-1, keepdim=True), min=self.min_norm)
            y_normalized = y / y_norm

            # Expand u if it's [B, D] or [D]
            if u.dim() == 2:
                u = u.unsqueeze(1).expand(-1, N, -1)  # [B, N, D]
            elif u.dim() == 1:
                u = u.view(1, 1, -1).expand(B, N, -1)

            v = torch.ones_like(x)
            v[:, :, 0:1] = -y_norm
            v[:, :, 1:] = (sqrtK - x0) * y_normalized

            alpha = torch.sum(y_normalized * u[:, :, 1:], dim=-1, keepdim=True) / sqrtK
            res = u - alpha * v
            return self.proj_tan(res, x, c)

        else:  # [N, D]
            N, D = x.size()
            x0 = x[:, 0:1]
            y = x[:, 1:]
            y_norm = torch.clamp(torch.norm(y, p=2, dim=-1, keepdim=True), min=self.min_norm)
            y_normalized = y / y_norm

            # Expand u if needed
            if u.dim() == 1:
                u = u.unsqueeze(0).expand(N, -1)  # [N, D]

            v = torch.ones_like(x)
            v[:, 0:1] = -y_norm
            v[:, 1:] = (sqrtK - x0) * y_normalized

            alpha = torch.sum(y_normalized * u[:, 1:], dim=-1, keepdim=True) / sqrtK
            res = u - alpha * v
            return self.proj_tan(res, x, c)

    def to_poincare(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)

