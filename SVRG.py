import math
import torch
from torch.optim.optimizer import Optimizer, required

class SVRG(Optimizer):
    r"""Stochastic Variance Reduced Gradient.
    """

    def __init__(self, params, lr=1e-2, weight_decay=0, epoch = 10, batch_size = None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.epoch = epoch
        self.batch_size = batch_size
        defaults = dict(lr = lr, weight_decay = weight_decay)
        super(SVRG, self).__init__(params, defaults)

        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['old_grad'] = torch.zeros_like(p.data)
                state['mu'] = torch.zeros_like(p.data)

    def save_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['old_grads'] = p.grad.data.clone()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is None:
            raise ValueError("Invalid closure function")

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['mu'] = p.grad.data.clone()

        for t in range(self.epoch):
            closure()
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('SVRD does not support sparse gradients')
                    state = self.state[p]
                    if group['weight_decay'] != 0:
                        grad = grad.add(group['weight_decay'], p.data)
                    p.data = p.data - group['lr'] * ( grad - state['old_grad'] + state['mu'])
        # return loss