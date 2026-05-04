"""
## Part 3: Simplified MHLA

**Background:** Standard attention caches both K and V for all heads, which uses
a lot of memory. MHLA compresses K and V into a lower-dimensional "latent" space.

**Standard attention cache per token:** $2 \times n_{\text{heads}} \times d_{\text{head}}$

**MHLA cache per token:** $d_{\text{latent}}$ (much smaller!)

**Your task:** Implement a simplified single-head version of MHLA.
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MHLA(nn.Module):

    def __init__(self, d_model: int = 256, d_latent: int = 64):
        """
        Simplified Multi-Head Latent Attention (single head version)

        Args:
            d_model: Model dimension
            d_latent: Latent dimension (compressed, this is what gets cached)
        """
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.scale = math.sqrt(d_latent)

        # TODO: Assign to instance attribute `linear_transformation_from_input_to_latent_input`
        # a linear transformation from an input with dimension `d_model` to a latent input with dimension `d_latent`
        # with no bias.
        self.linear_transformation_from_input_to_latent_input = nn.Linear(d_model, d_latent, bias=False)

        # TODO: Assign to instance attribute `linear_transformation_from_input_to_query`
        # a linear transformation from an input with dimension `d_model` to a query with dimension `d_model`
        # with no bias.
        self.linear_transformation_from_input_to_query = nn.Linear(d_model, d_model, bias=False)

        # TODO: Assign to instance attribute `linear_transformation_from_query_to_latent_query`
        # a linear transformation from a query with dimension `d_model` to a latent query with dimension `d_latent`
        # with no bias.
        self.linear_transformation_from_query_to_latent_query = nn.Linear(d_model, d_latent, bias=False)

        # TODO: Assign to instance attribute `linear_transformation_from_weighted_latent_input_to_output`
        # a linear transformation from a weighted latent input with dimension `d_latent` to an output with dimension `d_model`
        # with no bias.
        self.linear_transformation_from_weighted_latent_input_to_output = nn.Linear(d_latent, d_model, bias=False)

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input of shape (batch, seq_len, d_model)
            cache: Optional cached L_KV from previous steps
        Returns:
            output: (batch, seq_len, d_model)
            L_KV: (batch, total_seq_len, d_latent) for caching
        """
        batch, seq_len, _ = x.shape

        # TODO: Assign to local variable `tensor_of_latent_inputs`
        # the output of transforming the provided tensor of inputs to a tensor of latent inputs.
        tensor_of_latent_inputs = self.linear_transformation_from_input_to_latent_input(x)

        # TODO: Assign to local variable `tensor_of_queries`
        # the output of transforming the provided tensor of inputs to a tensor of queries.
        tensor_of_queries = self.linear_transformation_from_input_to_query(x)

        # TODO: Assign to local variable `tensor_of_latent_queries`
        # the output of transforming the provided tensor of queries to a tensor of latent queries.
        tensor_of_latent_queries = self.linear_transformation_from_query_to_latent_query(tensor_of_queries)

        # Step 3: Handle cache (for autoregressive generation)
        if cache is not None:
            # Concatenate with previous L_KV
            cached_tensor_of_latent_inputs = torch.cat([cache, tensor_of_latent_inputs], dim=1)
        else:
            cached_tensor_of_latent_inputs = tensor_of_latent_inputs

        # TODO: Assign to local variable `transposed_cached_tensor_of_latent_inputs`
        # the output of transposing the second to last and last dimensions of the cached tensor of latent inputs.
        transposed_cached_tensor_of_latent_inputs = cached_tensor_of_latent_inputs.transpose(-2, -1)

        # TODO: Assign to local variable `product` the product of the tensor of latent queries and
        # the transposed cached tensor of latent inputs.
        product = tensor_of_latent_queries @ transposed_cached_tensor_of_latent_inputs

        # TODO: Assign to local variable `tensor_of_attention_scores` the quotient of the product and the provided scale.
        tensor_of_attention_scores = product / self.scale

        # TODO: Assign to local variable `tensor_of_attention_weights` the output of applying softmax
        # to the tensor of attention scores along the last dimension.
        # raise NotImplementedError
        tensor_of_attention_weights = F.softmax(tensor_of_attention_scores, dim=-1)

        # TODO: Assign to local variable `tensor_of_weighted_latent_inputs`
        # the product of the tensor of attention weights and the cached tensor of latent inputs.
        # raise NotImplementedError
        tensor_of_weighted_latent_inputs = tensor_of_attention_weights @ cached_tensor_of_latent_inputs
        
        # TODO: Assign to local variable `tensor_of_outputs`
        # the output of transforming the tensor of weighted latent inputs to a tensor of outputs.
        # raise NotImplementedError
        tensor_of_outputs = self.linear_transformation_from_weighted_latent_input_to_output(tensor_of_weighted_latent_inputs)

        return tensor_of_outputs, cached_tensor_of_latent_inputs