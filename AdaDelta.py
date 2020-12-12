import torch
from torch.optim.optimizer import Optimizer, required

class AdaDelta(Optimizer):
    r"""Adadelta algorithm.
        $$
        \left\{
        \begin{aligned}
            g_t \gets & \frac{ \partial L(\theta_t) }{ \partial \theta } \\
            E[g^2]_t \gets & \gamma E[g^2]_{t-1} + (1 - \gamma) (g_t)^2 \\
            \Delta \theta_t \gets & \frac{\eta}{\sqrt{ E[g^2]_t +\epsilon }} g_t \\
            \theta_{t+1} \gets &  \theta_t - \Delta \theta_t
        \end{aligned}
        \right.
        $$
    """

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho <= 1.0:
            raise ValueError("Invalid rho value: {}".format(rho))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super(AdaDelta, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['square_avg'] = torch.zeros_like(p.data)
                if group['lr'] == 1:
                    state['acc_delta'] = torch.zeros_like(p.data)

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
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adadelta does not support sparse gradients')
                state = self.state[p]


                rho, eps = group['rho'], group['eps']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                state['square_avg'].mul_(rho).addcmul_(1 - rho, grad, grad)
                std = state['square_avg'].add(eps).sqrt_()

                if group['lr'] == 1:
                    delta = state['acc_delta'].add(eps).sqrt_().div_(std).mul_(grad)
                    state['acc_delta'].mul_(rho).addcmul_(1 - rho, delta, delta)
                else:
                    delta = 1 / std * grad #1.div_(std).mul_(grad)

                p.data.add_(-group['lr'], delta)

        return loss