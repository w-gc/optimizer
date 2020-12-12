import math
import random
import torch
from torch.optim.optimizer import Optimizer, required

class SAG(Optimizer):
    r"""Stochastic Average Gradient.
    """

    def __init__(self, params, lr=1e-2, old_grad_N = 10, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.old_grad_N = old_grad_N
        defaults = dict(lr = lr, weight_decay = weight_decay)
        super(SAG, self).__init__(params, defaults)

        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['old_grad'] = torch.zeros(torch.Size([self.old_grad_N]) + p.data.size())
                state['d'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        r = random.randrange(0, self.old_grad_N, 1)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError('ASGD does not support sparse gradients')
                state = self.state[p]

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                if(torch.cuda.is_available()):
                    state['d'] = state['d'] - state['old_grad'][r].cuda() + grad
                else:
                    state['d'] = state['d'] - state['old_grad'][r] + grad

                state['old_grad'][r] = grad
                p.data = p.data - group['lr'] / self.old_grad_N * state['d'] 
                
        return loss