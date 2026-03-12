import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def forward(self, decoder_hidden, encoder_outputs, src_mask = None):

        # TODO: Assign to local variable `tensor_of_current_decoder_hidden_states`
        # the output of converting the provided tensor of current decoder hidden states
        # from (batch size, number of hidden states) to (batch size, number of hidden states, 1).
        raise NotImplementedError

        # TODO: Assign to local variable `tensor_of_alignment_scores`
        # the output of batch matrix multiplying the provided tensor of encoder hidden states
        # and the tensor of current decoder hidden states.
        raise NotImplementedError
        
        # TODO: Assign to local variable `matrix_of_alignment_scores`
        # the output of converting the tensor of alignment scores from (batch size, number of time steps, 1)
        # to (batch size, number of time steps).
        raise NotImplementedError

        # TODO: If the provided mask describing which tokens are real and which are padding exists,
        #   reassign to `matrix_of_alignment_scores` the output of filling the matrix of alignment scores with `-1e9`,
        #   a large negative number, when corresponding values of the mask are false.
        raise NotImplementedError
        
        # TODO: Assign to local variable `matrix_of_attention_weights`
        # the output of passing the matrix of alignment scores to the softmax function from module functional.
        # Compute softmax along dimension 1.
        raise NotImplementedError

        # TODO: Assign to local variable `tensor_of_attention_weights`
        # the output of converting the provided matrix of attention weights
        # from (batch size, number of time steps) to (batch size, 1, number of time steps).
        raise NotImplementedError

        # TODO: Assign to local variable `context_tensor` the output of batch matrix multiplying
        # the tensor of attention weights and the tensor of encoder hidden states.
        raise NotImplementedError
        
        # TODO: Assign to local variable `context_matrix` the output of converting the context tensor
        # from (batch size, 1, number of hidden states) to (batch size, number of hidden states).
        raise NotImplementedError

        # TODO: Return the context matrix and matrix of attention weights.
        raise NotImplementedError