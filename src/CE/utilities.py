import torch
import torch.nn as nn
from torch.autograd import Function


class ReLUPlusFunc(Function):
    """
    ReLU+ (ReLUPlus) is a custom autograd function that behaves exactly like ReLU
    in the forward pass (clamps negatives to zero), but uses a leaky-style gradient
    in the backward pass to preserve recoverability from negative inputs. It
    has been statistically verified to be equivalent or faster on MNIST tasks.

    This ensures:
    - Clean hard gating behavior during inference.
    - Dead units during training can still receive gradient signal and recover.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, negative_slope: float):
        # Save input for use in backward; store slope in context.
        ctx.save_for_backward(input)
        ctx.negative_slope = negative_slope
        # Standard ReLU forward behavior: hard clamp below zero.
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Retrieve saved input and slope from forward
        input, = ctx.saved_tensors
        slope = ctx.negative_slope
        # Apply leaky-style gradient only where input was clamped (input <= 0)
        grad_input = torch.where(input <= 0, grad_output * slope, grad_output)
        return grad_input, None  # No gradient for the slope itself

class ReLUPlus(nn.Module):
    """
    A ReLUPlus module behaves exactly the same as
    a ReLU unit during inference, but is able to recover from an
    off situation when training. During backpropogation gradients
    propogate as though the unit was a leaky ReLU, allowing the
    model to turn dead neurons back on. This makes it suitable for
    gate behavior as well.
    """
    def __init__(self, leak_slope: float = 0.01):
        super().__init__()
        self.leak_slope = leak_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ReLUPlusFunc.apply(x, self.leak_slope)

class LogitDropout(nn.Module):
    """
    Dropout for logits. Sets dropped elements to -inf so they are ignored by softmax
    or similar activation functions. Guarantees at least one valid logit along a
    specified dimension.
    """
    def __init__(self, p: float = 0.5, comparison_dim: int = -1):
        """
        :param p: Dropout probability.
        :param comparison_dim: Dimension along which to ensure at least one element is kept.
        """
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError("Dropout probability must be in [0, 1]")
        self.p = p
        self.comparison_dim = comparison_dim

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return logits

        while True:
            keep_mask = torch.rand_like(logits) > self.p
            # Ensure at least one True per slice along comparison_dim
            if keep_mask.any(dim=self.comparison_dim).all():
                break

        return logits.masked_fill(~keep_mask, float('-inf'))

