import torch
from torch.optim.optimizer import Optimizer, required

class AdaGrad(Optimizer):
    r"""AdaGrad algorithm.
        $$
        \left\{
        \begin{aligned}
            g_t \gets & \frac{ \partial L(\theta_t) }{ \partial \theta } \\
            V_t \gets & \sqrt{\sum_{i = 0}^{t} \left(g_t\right)^2 + \epsilon} \\
            \theta \gets &  \theta - \frac{\eta}{V_t} g_t
        \end{aligned}
        \right.
        $$
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)
        super(AdaGrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.full_like(p.data, initial_accumulator_value)

    def __setstate__(self, state):
        super(AdaGrad, self).__setstate__(state)

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
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                clr = group['lr'] / ( 1 + (state['step'] - 1) * group['lr_decay'] )
                state['sum'].addcmul_(1, grad, grad)
                std = state['sum'].sqrt().add_(group['eps'])
                p.data.addcdiv_(-clr, grad, std)
        return loss