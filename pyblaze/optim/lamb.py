import math
import torch
import torch.optim as optim

class LAMB(optim.Optimizer):
    """
    Optimizer presented in "Large Batch Optimization for Deep Learning: Training Bert in 76
    Minutes" (You et al., 2019).

    The LAMB optimizer ("Layer-wise Adaptive Moments optimizer for Batch training") enables
    training on very large batches and provides an alternative for Adam whose performance
    deteriorates for large batches.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, lr_decay=0):
        """
        Initializes a new LAMB optimizer.

        Parameters
        ----------
        params: iterable of torch.Tensor or dict of str -> torch.Tensor
            The parameters to optimize, optionally overriding default values for parameter groups.
        lr: float, default: 1e-3
            The learning rate.
        betas: tuple of (float, float), default: (0.9, 0.999)
            The betas used to compute the running average of gradients.
        eps: float, default: 1e-8
            Epsilon parameter for numerical stability.
        weight_decay: float, default: 0
            L2 penalty to apply.
        lr_decay: float, default: 0
            Learning rate decay over each update.
        """
        assert lr > 0, "Learning rate must be greater 0."
        assert betas[0] > 0 and betas[1] > 0, "Beta values must be greater 0."
        assert betas[0] < 1 and betas[1] < 1, "Beta values must be smaller 1."
        assert eps > 0, "Epsilon must be gerater 0."

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            lr_decay=lr_decay
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure: callable, default: None
            A closure that reevaluates the model and returns the loss.
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
                    raise RuntimeError(
                        "LAMB does not support sparse gradients. " +
                        "Consider using SparseAdam instead."
                    )

                state = self.state[p]

                # Initialize state if needed
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Get relevant parameters
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Next step
                state['step'] += 1

                # Update learning rate if needed
                if group['lr_decay'] > 0:
                    decay = group['lr_decay']
                    lr = group['lr'] * (1 / 1 + decay * state['step'])
                else:
                    lr = group['lr']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                adam_step = exp_avg / denom

                # L2 penalty
                if group['weight_decay'] != 0:
                    adam_step.add_(group['weight_decay'], p.data)

                # Compute trust ratio and cap at 10
                r1 = p.data.norm()
                r2 = adam_step.norm()
                trust_ratio = min(r1 / r2, 10) if r1 != 0 and r2 != 0 else 1

                # Update weights
                p.data.add_(-step_size * trust_ratio, adam_step)

        return loss
        