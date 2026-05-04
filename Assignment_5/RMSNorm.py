"""
## Part 1: RMSNorm - Root Mean Square Normalization

**Background:** LayerNorm centers (zero mean) and scales (unit variance).
RMSNorm only scales, which is simpler and faster while being equally effective in Pre-LN architectures.
**Your task:** Complete the RMSNorm implementation below.
"""


import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: Feature dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps

        # TODO: Assign to local variable `tensor_of_1s` a tensor of a number of a 1s equal to the provided dimension.
        # raise NotImplementedError
        tensor_of_1s = torch.ones(dim)

        # TODO: Assign to instance variable scale a parameter based on the tensor of 1s.
        # raise NotImplementedError
        self.scale = nn.Parameter(tensor_of_1s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Normalized tensor of same shape
        """

        # TODO: Assign to local variable `tensor_of_squares` a tensor of squares of the elements in the provided input tensor.
        # raise NotImplementedError
        tensor_of_squares = x ** 2

        # TODO: Assign to local variable `tensor_of_means`
        # a tensor of the means of the tensor of squares along the last dimension.
        # Ensure that the number of dimensions of the tensor of means
        # is equal to the number of dimensions of the tensor of squares.
        # raise NotImplementedError
        tensor_of_means = tensor_of_squares.mean(dim=-1, keepdim=True)

        # TODO: Reassign to the tensor of means the sum of the tensor of means and the small constant.
        # raise NotImplementedError
        tensor_of_means = tensor_of_means + self.eps

        # TODO: Assign to local variable `root_mean_square` the square root of the tensor of means.
        # raise NotImplementedError
        root_mean_square = torch.sqrt(tensor_of_means)

        # TODO: Assign to local variable `normalized_input_tensor` the quotient of the input tensor and the root mean square.
        # raise NotImplementedError
        normalized_input_tensor = x / root_mean_square

        # TODO: Return the product of the scale and the normalized input tensor.
        # raise NotImplementedError
        return self.scale * normalized_input_tensor