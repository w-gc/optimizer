import math
import torch
from torch.optim.optimizer import Optimizer, required


class Adam(Optimizer):
    r"""Adam algorithm.
        $$
        \left\{
        \begin{aligned}
            g_t \gets & \frac{ \partial L(\theta_t) }{ \partial \theta } \\
            m_t \gets & \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
            v_t \gets & \beta_2 v_{t-1} + (1 - \beta_2) (g_t)^2 \\
            \hat{m}_t \gets & \frac{m_t}{1 - (\beta_1)^t} \\
            \hat{v}_t \gets & \frac{v_t}{1 - (\beta_2)^t} \\
            \theta \gets & \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t.
        \end{aligned}
        \right.
        $$
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)

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
                    raise RuntimeError('Adam does not support sparse gradients')
                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                state['exp_avg'].mul_(beta1).add_(1 - beta1, grad)
                state['exp_avg_sq'].mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                exp_avg_hat =  state['exp_avg'] / bias_correction1

                p.data.addcdiv_(-group['lr'], exp_avg_hat, denom)
        return loss