import torch
from torch.optim.optimizer import Optimizer, required

class MirrorDescent(Optimizer):
    r"""
    (batch, mini-batch)Stochastic gradient descent:
        $$
        \theta \gets \theta - \eta \sum_{i=1}^{B} \frac{\partial l_i}{\partial \theta}.
        $$
    """
    def __init__(self, params, lr=required, weight_decay=0, BreDivFun ='Squared norm'):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(MirrorDescent, self).__init__(params, defaults)
        self.BreDivFun = BreDivFun

    def __setstate__(self, state):
        super(MirrorDescent, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('nesterov', False)

    def _search_param(self, param, param_lr_grad, BreDivFun='Squared norm', norm_p = 0):
        ''' 
        $$
        \begin{aligned}
            x_{k + 1} &= \argmin_{x \in C} \left\{ f(x_k) + \left< g_k, x - x_k \right> + \frac{1}{\alpha_k} \text{Div}_{\psi}(x, x_k) \right\} \\
            &= \argmin_{x \in C} \left\{ \alpha_k f(x_k) + \alpha_k \left< g_k, x - x_k \right> + \text{Div}_{\psi}(x, x_k)  \right\} \\
            &= \argmin_{x \in C} \left\{ \left< \alpha_k g_k, x \right> + \text{Div}_{\psi}(x, x_k)  \right\}. \\
            &= \argmin_{x \in C} \left\{ \left< \alpha_k g_k, x \right> + \psi(x) - \psi(x_k) - \left< \nabla \psi (x_k), x - x_k \right>  \right\}. \\
            &= \argmin_{x \in C} \left\{ \left< \alpha_k g_k, x \right> + \psi(x) - \left< \nabla \psi (x_k), x \right>  \right\}. \\
            \Rightarrow & \alpha_k g_k + \nabla \psi (x) - \nabla \psi (x_k) |_{x = x_{k+1}} = 0. \\
        \end{aligned}
        $$
        '''
        
        if BreDivFun.strip() == 'Squared norm'.strip():
            # Squared norm: $\frac{1}{2}x^2$ , BreDiv: $\frac{1}{2}(x-y)^2$
            # \nabla \psi (x_k)  -  \alpha_k g_k
            new_param_data = param - param_lr_grad
            param = new_param_data
        elif BreDivFun.strip() == 'Shannon entropy'.strip():
            # Shannon entropy: $x \log x - x$, BreDiv: $x \log \frac{x}{y} - x +y$
            new_param_data = torch.log(param) - param_lr_grad
            param = torch.exp(new_param_data)
        elif BreDivFun.strip() == 'Burg entropy'.strip():
            # Burg entropy: $- \log x$ | $(0, +\infty)$ , BreDiv: $\frac{x}{y} - \log \frac{x}{y} - 1$
            new_param_data = torch.div(torch.ones_like(param), param) + param_lr_grad
            param = torch.div(torch.ones_like(param), new_param_data)
        # elif BreDivFun.strip() == 'Hellinger'.strip():
        #     # Hellinger: $- \sqrt{1 - x^2}$ , BreDiv: $(1 - xy)(1 - y^2)^{-\frac{1}{2}} - (1 - x^2 )^{\frac{1}{2}}$
        #     new_param_data = torch.mul(param, torch.rsqrt(torch.ones_like(param) - torch.pow(param, 2)) - param_lr_grad
        #     param = torch.mul(new_param_data, torch.rsqrt(torch.ones_like(new_param_data) + torch.pow(new_param_data, 2))
        elif BreDivFun.strip() == 'l_p quasi-norm'.strip() and p > 0 and p < 1:
            # l_p quasi-norm: $- x^p \quad (0<p<1)$ , BreDiv: $-x^p+pxy^{p-1}-(p-1)y^p$
            # if norm_p <= 0 or p >= 1:
            #     print('l_p quasi-norm: $- x^p$ and p must be (0<p<1)')
            #     break
            new_param_data = torch.pow(param, p-1) + torch.mul(1/p, param_lr_grad)
            param = torch.pow(new_param_data, 1 / (p-1))
        elif BreDivFun.strip() == 'l_p norm'.strip() and p > 1:
            # l_p norm: $- \vert x \vert^p \quad (1<p<\infty)$, BreDiv: $\vert x \vert^p - p x y^{p-1} \text{sgn}(y) + (p-1) \vert y \vert^p$
            # if norm_p <= 1:
            #     print('l_p norm: $- |x|^p$ and p must be (1<p<\infty)')
            #     break
            new_param_data = torch.mul(torch.pow(torch.abs(param), p-1), torch.sign(param)) + torch.mul(1/p, param_lr_grad)
            param = torch.mul(torch.pow(new_param_data, 1 / (p-1)), torch.sign(param))
        elif BreDivFun.strip() == 'Exponential'.strip():
            # Exponential: $\exp(x)$ , BreDiv:  $\exp(x) - \left(x - y + 1 \right) \exp(y)$
            new_param_data = torch.exp(param)- param_lr_grad
            param = torch.log(new_param_data)
        elif BreDivFun.strip() == 'Inverse'.strip():
            # Inverse: $\frac{1}{x}$ , BreDiv:  $\frac{1}{x} + \frac{x}{y^2} - \frac{2}{y}$
            new_param_data = torch.div(torch.ones_like(param), torch.pow(param,2)) + param_lr_grad
            param = torch.div(torch.div(torch.ones_like(param), torch.pow(new_param_data,2)))
        else :
            print('DID NOT implement this function in Bregman Divergence.')
            print('You can choose: Squared norm, Shannon entropy, Burg entropy, Hellinger, l_p quasi-norm, l_p norm, Exponential, Inverse.')
        return param

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
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                d_p.mul_(group['lr'])
                p.data = self._search_param(param = p.data, param_lr_grad = d_p, BreDivFun=self.BreDivFun)
                # p.data.add_(-group['lr'], d_p)
        return loss