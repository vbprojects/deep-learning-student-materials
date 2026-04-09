import torch
import torch.nn as nn
from typing import Optional


"""
## Part 4: Complete Transformer Block

Now let's combine everything into a modern Transformer block.
"""

class Transformer(nn.Module):

    def __init__(self, class_RMSNorm, class_MHLA, d_model: int = 256, d_latent: int = 64, mlp_ratio: int = 4):
        """
        Modern Transformer block with:
        - Pre-LN architecture
        - RMSNorm
        - Simplified MHLA
        - Standard MLP

        Args:
            d_model: Model dimension
            d_latent: Latent dimension for MHLA
            mlp_ratio: MLP expansion ratio
        """
        super().__init__()

        # Normalization (Pre-LN style)
        self.RMSNorm_1 = class_RMSNorm(d_model)
        self.RMSNorm_2 = class_RMSNorm(d_model)

        # Attention
        self.MHLA = class_MHLA(d_model, d_latent)

        # MLP
        d_mlp = d_model * mlp_ratio
        self.MLP = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model)
        )

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None):
        """
        Args:
            x: Input of shape (batch, seq_len, d_model)
            cache: Optional cached L_KV
        Returns:
            output: (batch, seq_len, d_model)
            L_KV: Updated cache
        """

        # TODO: Assign to local variable `scaled_normalized_input_tensor_1`
        # the output of calling the first instance of `RMSNorm` with the provided input.
        raise NotImplementedError

        # TODO: Assign to local variables `tensor_of_outputs_of_MHLA` and `cached_tensor_of_latent_inputs`
        # the outputs of calling MHLA with the first scaled normalized input tensor and the provided cache.
        raise NotImplementedError

        # TODO: Assign to local variable intermediate the sum of the provided input and the tensor of outputs of MHLA.
        raise NotImplementedError

        # TODO: Assign to local variable `scaled_normalized_input_tensor_2`
        # the output of calling the second instance of `RMSNorm` with intermediate.
        raise NotImplementedError

        # TODO: Assign to local variable `tensor_of_outputs_of_MLP`
        # the output of calling MLP with the second scaled normalized input tensor.
        raise NotImplementedError

        # TODO: Assign to local variable `tensor_of_outputs` the sum of intermediate and the tensor of outputs of MLP.
        raise NotImplementedError

        return tensor_of_outputs, cached_tensor_of_latent_inputs