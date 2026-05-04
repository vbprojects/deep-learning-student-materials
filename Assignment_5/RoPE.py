import torch
import torch.nn as nn
from typing import cast


"""
## Part 2: RoPE - Rotary Positional Embeddings

**Background:** Instead of adding position information to embeddings, RoPE rotates
the query and key vectors based on their position. This encodes *relative* position
in the attention scores.

**Key insight:** After rotation, the dot product between $q_m$ and $k_n$ depends
only on the distance $(m-n)$, not absolute positions.

**Your task:** Implement RoPE following the steps below.
"""

class RoPE(nn.Module):

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        """
        Args:
            dim: Embedding dimension (must be even)
            max_seq_len: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even"

        # TODO: Assign to local variable `tensor_of_indices` a 1D tensor with values from the interval [0, provided dimension)
        # taken with common difference 2 beginning from 0.
        # raise NotImplementedError
        tensor_of_indices = torch.arange(0, dim, 2)

        # TODO: Reassign to the tensor of indices the tensor of floating point representations of the indices.
        # raise NotImplementedError
        tensor_of_indices = tensor_of_indices.float()

        # TODO: Assign to local variable `normalized_tensor_of_indices`
        # the quotient of the tensor of indices and the provided dimension.
        # raise NotImplementedError
        normalized_tensor_of_indices = tensor_of_indices / dim

        # TODO: Assign to local variable `tensor_of_exponents` the provided base raised to the normalized tensor of indices.
        # raise NotImplementedError
        tensor_of_exponents = base ** normalized_tensor_of_indices

        # TODO: Assign to local variable `tensor_of_inverse_frequencies` the reciprocal of the tensor of exponents.
        # raise NotImplementedError
        tensor_of_inverse_frequencies = 1.0 / tensor_of_exponents

        # TODO: Assign to local variable `tensor_of_positions`
        # a 1D tensor with values from the interval [0, provided maximum sequence length)
        # taken with common difference 1 beginning from 0.
        # raise NotImplementedError
        tensor_of_positions = torch.arange(0, max_seq_len, 1)
        # TODO: Reassign to the tensor of positions the tensor of floating point representations of the positions.
        # raise NotImplementedError
        tensor_of_positions = tensor_of_positions.float()

        # TODO: Assign to local variable `tensor_of_frequencies` the outer product of the tensor of positions and
        # the tensor of inverse frequencies.
        # raise NotImplementedError
        tensor_of_frequencies = torch.outer(tensor_of_positions, tensor_of_inverse_frequencies)

        # Register the tensor of frequencies as a buffer of this instance called `tensor_of_frequencies`.
        self.register_buffer("tensor_of_frequencies", tensor_of_frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embeddings to input.

        Args:
            x: Input of shape (batch, seq_len, dim)
        Returns:
            Rotated input of same shape
        """
        batch, seq_len, dim = x.shape

        # Get frequencies for this sequence length
        buffer = cast(torch.Tensor, self.tensor_of_frequencies)
        tensor_of_frequencies = buffer[:seq_len] # (seq_len, dim/2)

        # TODO: Assign to local variable `tensor_of_cosines` a tensor of the cosines of the frequencies.
        # raise NotImplementedError
        tensor_of_cosines = torch.cos(tensor_of_frequencies)

        # TODO: Assign to local variable `tensor_of_sines` a tensor of the sines of the frequencies.
        # raise NotImplementedError
        tensor_of_sines = torch.sin(tensor_of_frequencies)

        # Reshape x into pairs: (batch, seq_len, dim/2, 2)
        x_reshaped = x.reshape(batch, seq_len, -1, 2)

        # Split into even and odd indices
        tensor_of_inputs_with_even_indices = x_reshaped[..., 0]  # (batch, seq_len, dim/2)
        tensor_of_inputs_with_odd_indices = x_reshaped[..., 1]   # (batch, seq_len, dim/2)

        # TODO: Assign to local variable `tensor_of_rotated_inputs_with_even_indices`
        # the difference of the product of the tensor of inputs with even indices and the tensor of cosines and
        # the product of the tensor of inputs with odd indices and the tensor of sines. 
        # raise NotImplementedError
        tensor_of_rotated_inputs_with_even_indices = tensor_of_inputs_with_even_indices * tensor_of_cosines - tensor_of_inputs_with_odd_indices * tensor_of_sines

        # TODO: Assign to local variable `tensor_of_rotated_inputs_with_odd_indices`
        # the sum of the product of the tensor of inputs with even indices and the tensor of sines and
        # the product of the tensor of inputs with odd indices and the tensor of cosines. 
        # raise NotImplementedError
        tensor_of_rotated_inputs_with_odd_indices = tensor_of_inputs_with_even_indices * tensor_of_sines + tensor_of_inputs_with_odd_indices * tensor_of_cosines

        # Stack back together
        tensor_of_rotated_inputs = torch.stack([tensor_of_rotated_inputs_with_even_indices, tensor_of_rotated_inputs_with_odd_indices], dim = -1)

        # Reshape back to original shape
        return tensor_of_rotated_inputs.reshape(batch, seq_len, dim)