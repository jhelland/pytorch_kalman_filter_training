import torch
from torch.optim.optimizer import Optimizer, required

import numpy as np
import scipy.linalg

from functools import reduce


class EKF(Optimizer):
    def __init__(self, params, lr=required, num_outputs=required, P=None, Q=None, R=None, eps=1e-3):
        defaults = dict(lr=lr)
        super(EKF, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("EKF does not support per-parameter options "
                             "(parameter groups)")

        self._numel_cache = None
        self._params = self.param_groups[0]["params"]

        num_weights = self._numel()

        if P is None:
            self.P = 1.0 / eps * np.eye(num_weights)
        elif np.isscalar(P):
            self.P = P * np.eye(num_weights)
        else:
            self.P = P
        self.P = torch.from_numpy(self.P.astype("float32"))

        if Q is None:
            self.Q = np.zeros((num_weights, num_weights))
        elif np.isscalar(Q):
            self.Q = Q * np.eye(num_weights)
        else:
            self.Q = Q
        self.Q = torch.from_numpy(self.Q.astype("float32"))

        if R is None:
            self.R = 1e-1 * np.eye(num_outputs)
        elif np.isscalar(R):
            self.R = R * np.eye(num_outputs)
        else:
            self.R = self.R
        self.R = torch.from_numpy(self.R.astype("float32"))

        self.y = None  # target vector
        self.h = None  # network output for u
        self.num_outputs = num_outputs

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)

        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            raise ValueError("closure not implemented")

        assert (self.y is not None) and (self.h is not None)

        jacobian = torch.zeros(self.num_outputs, self._numel())
        grad_output = torch.zeros(self.num_outputs)
        for i in range(self.num_outputs):
            self.zero_grad()
            grad_output.zero_()
            grad_output[i] = 1.
            self.h.backward(grad_output, retain_graph=True)
            jacobian[i] = self._gather_flat_grad()

        group = self.param_groups[0]
        lr = group["lr"]

        # Global scaling matrix
        A = self.R + jacobian @ self.P @ jacobian.t()
        A.data = torch.from_numpy( scipy.linalg.pinvh(A.data) )

        # Kalman gain matrix
        K = self.P @ jacobian.t() @ A
        # residual between target and model output
        residual = self.y - self.h

        # Update state i.e. model weights
        self._add_grad(lr, K @ residual)

        # Update covariance estimate
        self.P = self.P - K @ jacobian @ self.P + self.Q

        return loss