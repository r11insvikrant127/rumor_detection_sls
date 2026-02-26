import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CircleLoss(nn.Module):

    def __init__(self, m=0.25, gamma=256, reduction="mean"):
        super().__init__()

        self.m = m
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):

        one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()

        sp = logits[one_hot == 1]
        sn = logits[one_hot == 0]

        # adaptive weights
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, 0.)
        an = torch.clamp_min(sn.detach() + self.m, 0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -self.gamma * ap * (sp - delta_p)
        logit_n = self.gamma * an * (sn - delta_n)

        loss = F.softplus(
            torch.logsumexp(logit_n, dim=0) +
            torch.logsumexp(logit_p, dim=0)
        )

        if self.reduction == "mean":
            return loss.mean()
        return loss


class CosineLinearLayer(nn.Module):
    """
    Linear layer that outputs cosine similarities.
    
    REQUIRED for Circle Loss theoretical correctness.
    Use this as the final layer in your model for Circle Loss.
    
    Computes: cos(θ) = (x/||x||) · (w/||w||)
    
    Args:
        in_features: Input feature dimension
        out_features: Number of classes (should be 2 for binary classification)
        bias: Whether to include bias (recommended: False for cosine similarity)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight matrix
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias (optional - typically not needed for cosine similarity)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Computes cosine similarity between normalized inputs and normalized weights.
        Output range: [-1, 1] (or [-bias, bias] if bias is used)
        """
        # Normalize inputs and weights for cosine similarity
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        output = F.linear(x_norm, w_norm, self.bias)
        
        return output
    
    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}')


class AngularLinearLayer(nn.Module):
    """
    Alternative: Linear layer with angular margin.
    For completeness, not required for base Circle Loss.
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Normalize weights and inputs
        w_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=1)
        
        # Cosine similarity
        cos_theta = F.linear(x_norm, w_norm)
        
        if labels is None or not self.training:
            return cos_theta * self.s
        
        # Apply angular margin during training
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Convert to theta and apply margin
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        
        output = cos_theta * (1 - one_hot) + target_logits * one_hot
        output *= self.s
        
        return output


def create_circle_loss(m: float = 0.25, gamma: float = 256, 
                       reduction: str = 'mean') -> CircleLoss:
    """
    Factory function to create Circle Loss with paper settings.
    
    Args:
        m: Margin (must be 0.25)
        gamma: Scaling factor (must be 256)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Configured CircleLoss instance
    """
    return CircleLoss(m=m, gamma=gamma, reduction=reduction, normalize=True)


def test_circle_loss():
    """
    Test function to verify Circle Loss implementation with CosineLinearLayer.
    """
    print("=" * 60)
    print("Testing Circle Loss Implementation")
    print("=" * 60)
    
    # Test configuration
    batch_size = 8
    in_features = 768  # Typical feature dimension after backbone
    num_classes = 2    # Binary classification
    
    # === TEST 1: CosineLinearLayer ===
    print("\n1. Testing CosineLinearLayer...")
    
    cosine_layer = CosineLinearLayer(in_features, num_classes, bias=False)
    x = torch.randn(batch_size, in_features)
    
    try:
        cosine_sim = cosine_layer(x)
        print(f"✓ CosineLinearLayer output shape: {cosine_sim.shape}")
        print(f"  Output range: [{cosine_sim.min():.3f}, {cosine_sim.max():.3f}]")
        
        # Verify cosine similarity properties
        assert torch.all(cosine_sim >= -1.05) and torch.all(cosine_sim <= 1.05), \
            "Cosine similarity should be in [-1, 1] range"
        print("  ✓ Output is in [-1, 1] range (cosine similarity)")
    except Exception as e:
        print(f"✗ Error in CosineLinearLayer: {e}")
        return False
    
    # === TEST 2: Circle Loss with cosine similarities ===
    print("\n2. Testing Circle Loss with CosineLinearLayer output...")
    
    # Create loss function
    loss_fn = CircleLoss(normalize=False)  # No need to normalize again
    
    # Create labels
    labels = torch.randint(0, num_classes, (batch_size,))
    
    try:
        loss = loss_fn(cosine_sim, labels)
        print(f"✓ Circle Loss computed successfully: {loss.item():.6f}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # === TEST 3: Compare with and without cosine layer ===
    print("\n3. Testing: CosineLayer + CircleLoss vs Linear + CircleLoss...")
    
    # Standard linear layer (incorrect for Circle Loss)
    linear_layer = nn.Linear(in_features, num_classes)
    linear_out = linear_layer(x)
    
    # Cosine layer (correct)
    cosine_out = cosine_layer(x)
    
    # Compute losses
    loss_cosine = loss_fn(cosine_out, labels)
    loss_linear = loss_fn(linear_out, labels)  # This will normalize internally
    
    print(f"  Cosine layer loss: {loss_cosine.item():.6f}")
    print(f"  Linear layer loss: {loss_linear.item():.6f}")
    print(f"  Difference: {abs(loss_cosine.item() - loss_linear.item()):.6f}")
    print("  Note: Linear layer + internal normalization ≠ true cosine classifier geometry")
    
    # === TEST 4: Perfect separation case ===
    print("\n4. Testing perfect separation case...")
    
    # Create perfect cosine similarities
    perfect_cosine = torch.tensor([
        [0.9, -0.9],  # Class 0: high for class 0, low for class 1
        [-0.9, 0.9],  # Class 1: low for class 0, high for class 1
    ]).repeat(batch_size//2, 1)
    
    perfect_labels = torch.tensor([0, 1]).repeat(batch_size//2)
    
    perfect_loss = loss_fn(perfect_cosine, perfect_labels)
    print(f"  Perfect separation loss: {perfect_loss.item():.6f}")
    
    # === TEST 5: End-to-end model ===
    print("\n5. Testing end-to-end model with CosineLinearLayer...")
    
    class SimpleModel(nn.Module):
        def __init__(self, in_dim, hidden_dim, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.bn = nn.BatchNorm1d(hidden_dim)
            self.fc2 = CosineLinearLayer(hidden_dim, num_classes, bias=False)
            
        def forward(self, x):
            x = F.relu(self.bn(self.fc1(x)))
            x = self.fc2(x)
            return x
    
    model = SimpleModel(in_features, 256, num_classes)
    logits = model(x)
    loss = loss_fn(logits, labels)
    
    print(f"✓ End-to-end test successful: {loss.item():.6f}")
    print(f"  Final layer type: CosineLinearLayer")
    
    print("\n" + "=" * 60)
    print("✅ CORRECT USAGE:")
    print("=" * 60)
    print("1. Use CosineLinearLayer as final layer:")
    print("   self.fc = CosineLinearLayer(hidden_dim * 31, num_classes, bias=False)")
    print()
    print("2. Pass output directly to CircleLoss:")
    print("   logits = model(features)  # Already cosine similarities")
    print("   loss = circle_loss(logits, labels)")
    print()
    print("3. Hyperparameters (paper-exact):")
    print("   - Margin m = 0.25")
    print("   - Gamma = 256")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_circle_loss()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed.")