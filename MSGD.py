import torch
from torch.optim.optimizer import Optimizer, required

class MSGD(Optimizer):
    r"""
    Momentum Stochastic gradient descent:
        $$
        \left\{
        \begin{aligned}
        v \gets &  \gamma v + \eta \frac{\partial L}{\partial \theta} \\
        \theta \gets &  \theta - v.
        \end{aligned}
        \right.
        $$
    """

    def __init__(self, params, lr=required, momentum=0.9, dampening=0, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)
        super(MSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MSGD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['weight_decay'] != 0:
                    d_p.add_(group['weight_decay'], p.data)
                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        param_state['momentum_buffer'].mul_(group['momentum']).add_(1 - group['dampening'], d_p)
                    d_p = param_state['momentum_buffer']
                p.data.add_(-group['lr'], d_p)
        return loss
