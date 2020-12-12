import torch
from torch.optim.optimizer import Optimizer, required

class RMSProp(Optimizer):
    r"""RMSprop algorithm.
        $$
        \left\{
        \begin{aligned}
            g_t \gets & \frac{ \partial L(\theta_t) }{ \partial \theta } \\
            E[g^2]_t \gets & 0.9 E[g^2]_{t-1} + 0.1 (g_t)^2 \\
            \Delta \theta_t \gets & \frac{\eta}{\sqrt{ E[g^2]_t +\epsilon }} g_t \\
            \theta_{t+1} \gets &  \theta_t - \Delta \theta_t
        \end{aligned}
        \right.
        $$
    """

    def __init__(self, params, lr=1e-2, eps=1e-6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super(RMSProp, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['square_avg'] = torch.zeros_like(p.data)

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
                    raise RuntimeError('RMSProp does not support sparse gradients')
                state = self.state[p]
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                state['square_avg'].mul_(0.9).addcmul_(0.1, grad, grad)
                std = state['square_avg'].add(group['eps']).sqrt_()
                delta = 1 / std * grad 
                p.data.add_(-group['lr'], delta)
        return loss