from importlib import import_module


"""
## Setup and Installation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import matplotlib.pyplot as plt
import time
import numpy as np
from typing import Optional, Tuple


def run(RMSNorm, RoPE, MHLA, Transformer) -> None:

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print("Setup complete!")


    ## Part 1: RMSNorm (20 points)

    # Test your implementation
    def test_rmsnorm():
        """Test that RMSNorm produces correct scale"""
        print("Testing RMSNorm...")
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64)
        out = norm(x)

        # Check 1: Output should have RMS ≈ 1
        rms = torch.sqrt((out ** 2).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5), "RMS should be ~1"
        print(f"  ✓ Output RMS: {rms.mean().item():.6f} (should be ~1.0)")

        # Check 2: Compare with LayerNorm - RMSNorm mean is NOT forced to zero
        layer_norm = nn.LayerNorm(64)
        out_ln = layer_norm(x)

        mean_rms = out.mean(dim=-1).abs().mean()
        mean_ln = out_ln.mean(dim=-1).abs().mean()

        print(f"  ✓ RMSNorm output mean: {mean_rms.item():.4f}")
        print(f"  ✓ LayerNorm output mean: {mean_ln.item():.6f} (near zero)")
        print(f"  ✓ RMSNorm does NOT center data (unlike LayerNorm)")

        print("✓ RMSNorm test passed!")
        return out

    out = test_rmsnorm()
    print(out)


    """
    ### Comparison: RMSNorm vs LayerNorm

    Let's compare the two normalization methods.
    """

    def compare_normalizations():
        """Compare RMSNorm and LayerNorm"""
        dim = 512
        batch, seq_len = 4, 128

        # Create input
        x = torch.randn(batch, seq_len, dim).to(device)

        # Initialize both norms
        rms_norm = RMSNorm(dim).to(device)
        layer_norm = nn.LayerNorm(dim).to(device)

        # Apply both
        out_rms = rms_norm(x)
        out_ln = layer_norm(x)

        # Compare statistics
        print("RMSNorm output - Mean: {:.4f}, Std: {:.4f}".format(
            out_rms.mean().item(), out_rms.std().item()))
        print("LayerNorm output - Mean: {:.4f}, Std: {:.4f}".format(
            out_ln.mean().item(), out_ln.std().item()))

        # Speed comparison
        n_iters = 1000

        # Warmup
        for _ in range(10):
            _ = rms_norm(x)
            _ = layer_norm(x)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # RMSNorm timing
        start = time.time()
        for _ in range(n_iters):
            _ = rms_norm(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        rms_time = time.time() - start

        # LayerNorm timing
        start = time.time()
        for _ in range(n_iters):
            _ = layer_norm(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        ln_time = time.time() - start

        print(f"\nSpeed comparison ({n_iters} iterations):")
        print(f"RMSNorm: {rms_time:.4f}s")
        print(f"LayerNorm: {ln_time:.4f}s")
        if ln_time > rms_time:
            print(f"Speedup: {ln_time/rms_time:.2f}x")
        else:
            print(f"Note: On CPU, PyTorch's optimized LayerNorm may be faster.")
            print(f"      RMSNorm shows speedups on GPU or with optimized kernels.")

        # Memory comparison
        print(f"\nParameter count:")
        print(f"RMSNorm: {sum(p.numel() for p in rms_norm.parameters())} (only gamma)")
        print(f"LayerNorm: {sum(p.numel() for p in layer_norm.parameters())} (gamma + beta)")

    compare_normalizations()


    # Test RoPE
    def test_rope():
        """Test that RoPE encodes relative positions"""
        print("Testing RoPE...")
        rope = RoPE(dim=64)

        # Create separate Q and K vectors (same content, will be rotated differently)
        torch.manual_seed(123)  # For reproducibility
        base_vec = torch.randn(1, 1, 64)  # Single base vector

        # Create Q and K at different positions by repeating the base vector
        q_positions = torch.tensor([0, 10])  # Query at positions 0 and 10
        k_positions = torch.tensor([5, 15])  # Key at positions 5 and 15

        # Create sequences where we place our base vector at specific positions
        seq_len = 20
        Q = torch.zeros(1, seq_len, 64)
        K = torch.zeros(1, seq_len, 64)

        # Place the same base vector at different positions
        Q[0, q_positions[0]] = base_vec[0, 0]
        Q[0, q_positions[1]] = base_vec[0, 0]
        K[0, k_positions[0]] = base_vec[0, 0]
        K[0, k_positions[1]] = base_vec[0, 0]

        # Apply RoPE
        Q_rot = rope(Q)
        K_rot = rope(K)

        # Test relative position property
        # dot(q@pos0, k@pos5) should equal dot(q@pos10, k@pos15) (both have distance 5)
        dot1 = (Q_rot[0, q_positions[0]] * K_rot[0, k_positions[0]]).sum()
        dot2 = (Q_rot[0, q_positions[1]] * K_rot[0, k_positions[1]]).sum()

        print(f"  Dot product at distance 5 (positions 0→5): {dot1:.4f}")
        print(f"  Dot product at distance 5 (positions 10→15): {dot2:.4f}")
        print(f"  Difference: {(dot1 - dot2).abs():.6f}")

        assert torch.allclose(dot1, dot2, atol=1e-3), "Should encode relative position!"
        print("✓ RoPE relative position test passed!")

        # Test length extrapolation
        print("\nTesting length extrapolation...")
        x_long = torch.randn(1, 100, 64)
        x_long_rotated = rope(x_long)
        print(f"  ✓ Can process sequences longer than some typical lengths ({x_long.shape[1]} tokens)")

    test_rope()


    """
    ### Visualize RoPE

    Let's visualize how RoPE encodes positions.
    """

    def visualize_rope():
        """Visualize RoPE attention patterns"""
        rope = RoPE(dim=64)

        # Create queries and keys at different positions
        seq_len = 50
        x = torch.randn(1, seq_len, 64)
        x_rotated = rope(x)

        # Compute attention scores (without softmax)
        scores = torch.matmul(x_rotated, x_rotated.transpose(-2, -1))
        scores = scores[0].detach().numpy()  # (seq_len, seq_len)

        # Plot
        plt.figure(figsize=(12, 4))

        # Full attention matrix
        plt.subplot(1, 3, 1)
        plt.imshow(scores, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Score')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title('Attention Scores with RoPE')

        # Attention as function of relative distance
        plt.subplot(1, 3, 2)
        distances = []
        avg_scores = []
        for d in range(25):
            # Get all pairs with distance d
            mask = torch.zeros(seq_len, seq_len)
            for i in range(seq_len - d):
                mask[i, i + d] = 1

            if mask.sum() > 0:
                avg_score = (torch.tensor(scores) * mask).sum() / mask.sum()
                distances.append(d)
                avg_scores.append(avg_score.item())

        plt.plot(distances, avg_scores, marker='o', linewidth=2)
        plt.xlabel('Relative Distance')
        plt.ylabel('Average Attention Score')
        plt.title('Attention vs Relative Position')
        plt.grid(True, alpha=0.3)

        # Show that position 0 has same pattern as position 20
        plt.subplot(1, 3, 3)
        plt.plot(scores[0, :], label='Query at position 0', linewidth=2)
        plt.plot(scores[20, :], label='Query at position 20', linewidth=2, linestyle='--')
        plt.xlabel('Key Position')
        plt.ylabel('Attention Score')
        plt.title('RoPE Relative Position Property')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("visualization_of_RoPE.png")
        plt.show()

        print("Notice: The patterns are shifted but have the same shape!")
        print("This shows that RoPE encodes relative position, not absolute position.")

    visualize_rope()

    # Test MHLA
    def test_mhla():
        """Test SimplifiedLatentAttention"""
        print("Testing MHLA...")

        d_model, d_latent = 256, 64
        mhla = MHLA(d_model, d_latent)

        # Test forward pass
        x = torch.randn(2, 10, d_model)
        output, L_KV = mhla(x)

        # Check shapes
        assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
        assert L_KV.shape == (2, 10, d_latent), f"L_KV shape mismatch: {L_KV.shape}"
        print(f"  ✓ Forward pass: input {x.shape} -> output {output.shape}")
        print(f"  ✓ Cache shape: {L_KV.shape}")

        # Test with cache
        x_next = torch.randn(2, 1, d_model)
        output_next, L_KV_next = mhla(x_next, cache=L_KV)

        assert output_next.shape == (2, 1, d_model), "Cached output shape wrong"
        assert L_KV_next.shape == (2, 11, d_latent), "Cached L_KV shape wrong"
        print(f"  ✓ With cache: new input {x_next.shape} -> cache {L_KV_next.shape}")

        print("✓ MHLA test passed!")

    test_mhla()

    """
    ### Standard Attention for Comparison

    Here's standard attention implemented for comparison (already complete).
    """

    class StandardAttention(nn.Module):
        def __init__(self, d_model: int = 256):
            super().__init__()
            self.d_model = d_model
            self.scale = math.sqrt(d_model)

            self.W_Q = nn.Linear(d_model, d_model, bias=False)
            self.W_K = nn.Linear(d_model, d_model, bias=False)
            self.W_V = nn.Linear(d_model, d_model, bias=False)
            self.W_O = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x: torch.Tensor, cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
            """
            Args:
                x: Input of shape (batch, seq_len, d_model)
                cache: Optional tuple of (cached_K, cached_V)
            Returns:
                output: (batch, seq_len, d_model)
                (K, V): Tuple for caching
            """
            Q = self.W_Q(x)
            K_new = self.W_K(x)
            V_new = self.W_V(x)

            # Handle cache
            if cache is not None:
                K_cache, V_cache = cache
                K = torch.cat([K_cache, K_new], dim=1)
                V = torch.cat([V_cache, V_new], dim=1)
            else:
                K, V = K_new, V_new

            # Attention
            scores = (Q @ K.transpose(-2, -1)) / self.scale
            attn_weights = F.softmax(scores, dim=-1)
            output = attn_weights @ V
            output = self.W_O(output)

            return output, (K, V)

    """
    ### Compare MHLA vs Standard Attention
    """

    def compare_attention_mechanisms():
        """Compare cache sizes and efficiency"""
        d_model = 256
        d_latent = 64
        seq_lengths = [50, 100, 200, 500, 1000, 2000]

        mhla = MHLA(d_model, d_latent)
        std_attn = StandardAttention(d_model)

        mhla_cache_sizes = []
        std_cache_sizes = []

        for seq_len in seq_lengths:
            # MHLA cache: only L_KV
            mhla_cache = seq_len * d_latent
            mhla_cache_sizes.append(mhla_cache)

            # Standard attention cache: both K and V
            std_cache = seq_len * d_model * 2
            std_cache_sizes.append(std_cache)

        # Plot
        plt.figure(figsize=(12, 4))

        # Cache size comparison
        plt.subplot(1, 2, 1)
        plt.plot(seq_lengths, np.array(std_cache_sizes)/1000, 'o-', label='Standard Attention', linewidth=2, markersize=8)
        plt.plot(seq_lengths, np.array(mhla_cache_sizes)/1000, 's-', label='MHLA', linewidth=2, markersize=8)
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('Cache Size (thousands of values)', fontsize=12)
        plt.title('KV Cache Size vs Sequence Length', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        # Compression ratio
        plt.subplot(1, 2, 2)
        ratios = [std / mhla for std, mhla in zip(std_cache_sizes, mhla_cache_sizes)]
        plt.plot(seq_lengths, ratios, 'o-', linewidth=2, markersize=8, color='green')
        plt.axhline(y=ratios[0], color='gray', linestyle='--', alpha=0.5, label=f'{ratios[0]:.1f}x (constant)')
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('Compression Ratio', fontsize=12)
        plt.title('Cache Compression: Standard / MHLA', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("visualization_of_MHLA.png")
        plt.show()

        print(f"\nFor d_model={d_model}, d_latent={d_latent}:")
        print(f"Compression ratio: {ratios[0]:.1f}x")
        print(f"At seq_len=2000: Standard={std_cache_sizes[-1]:,} vs MHLA={mhla_cache_sizes[-1]:,}")
        print(f"Memory savings: {(1 - mhla_cache_sizes[-1]/std_cache_sizes[-1])*100:.1f}%")

    compare_attention_mechanisms()

    # Test the complete block
    def test_transformer_block():
        """Test the complete modern transformer block"""
        print("Testing Modern Transformer Block...")

        block = Transformer(RMSNorm, MHLA, d_model=256, d_latent=64)

        # Forward pass
        x = torch.randn(2, 10, 256)
        output, cache = block(x)

        assert output.shape == x.shape, "Output shape mismatch"
        assert cache.shape == (2, 10, 64), "Cache shape mismatch"
        print(f"  ✓ Forward pass: {x.shape} -> {output.shape}")

        # Test with cache (autoregressive)
        x_next = torch.randn(2, 1, 256)
        output_next, cache_next = block(x_next, cache=cache)

        assert output_next.shape == (2, 1, 256), "Cached output shape wrong"
        assert cache_next.shape == (2, 11, 64), "Updated cache shape wrong"
        print(f"  ✓ With cache: {x_next.shape} -> cache {cache_next.shape}")

        print("✓ Transformer block test passed!")
        print(f"\nCache compression: Standard would cache {10 * 256 * 2} values,")
        print(f"MHLA only caches {10 * 64} values = {(10*256*2)/(10*64):.1f}x smaller!")

    test_transformer_block()

    """
    ### Simple Next-Token Prediction Task

    Let's test our modern transformer on a simple task.
    """

    def create_toy_dataset(vocab_size: int = 100, seq_len: int = 32, n_samples: int = 1000):
        """Create a simple synthetic dataset for next-token prediction"""
        # Simple pattern: next token = (current token + 1) mod vocab_size
        data = torch.randint(0, vocab_size, (n_samples, seq_len))
        targets = (data + 1) % vocab_size
        return data, targets

    def train_model(model, n_epochs: int = 100, vocab_size: int = 100):
        """Simple training loop"""
        d_model = 256

        # Dataset
        train_data, train_targets = create_toy_dataset(vocab_size=vocab_size, n_samples=800, seq_len=32)
        val_data, val_targets = create_toy_dataset(vocab_size=vocab_size, n_samples=200, seq_len=32)

        # Move to device
        train_data = train_data.to(device)
        train_targets = train_targets.to(device)
        val_data = val_data.to(device)
        val_targets = val_targets.to(device)

        # Embedding and output layers
        embedding = nn.Embedding(vocab_size, d_model).to(device)
        output_proj = nn.Linear(d_model, vocab_size).to(device)
        model = model.to(device)

        # Combine into simple model
        params = list(model.parameters()) + list(embedding.parameters()) + list(output_proj.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)

        train_losses = []
        val_losses = []

        print("Training model...")
        for epoch in range(n_epochs):
            # Training
            model.train()
            optimizer.zero_grad()

            # Forward pass
            x_embed = embedding(train_data)  # (batch, seq, d_model)
            output, _ = model(x_embed)
            logits = output_proj(output)  # (batch, seq, vocab)

            # Loss
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                train_targets.reshape(-1)
            )

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Validation
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    x_embed_val = embedding(val_data)
                    output_val, _ = model(x_embed_val)
                    logits_val = output_proj(output_val)
                    val_loss = F.cross_entropy(
                        logits_val.reshape(-1, vocab_size),
                        val_targets.reshape(-1)
                    )
                    val_losses.append(val_loss.item())

                    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        return train_losses, val_losses

    print("\n" + "="*60)
    print("Training modern transformer on toy task...")
    print("="*60)
    modern_model = Transformer(RMSNorm, MHLA, d_model=256, d_latent=64)
    train_losses, val_losses = train_model(modern_model, n_epochs=100)

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', alpha=0.7, linewidth=1)
    plt.plot([i*10-1 for i in range(1, len(val_losses)+1)], val_losses, 'o-', label='Val Loss', markersize=6, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Progress: Next-Token Prediction', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualization_of_Transformer.png")
    plt.show()

    print(f"\nFinal training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")


def main() -> None:
    RMSNorm = import_module("RMSNorm")
    RoPE = import_module("RoPE")
    MHLA = import_module("MHLA")
    Transformer = import_module("Transformer")
    run(RMSNorm.RMSNorm, RoPE.RoPE, MHLA.MHLA, Transformer.Transformer)


if __name__ == '__main__':
    main()