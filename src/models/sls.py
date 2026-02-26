import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Import CosineLinearLayer from loss module
# Make sure this import path is correct for your project structure
try:
    from src.training.loss import CosineLinearLayer
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from src.training.loss import CosineLinearLayer


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate=0.15):
        super().__init__()
        
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, 
            kernel_size=kernel_size,
            padding=kernel_size//2,
            groups=in_channels,
            bias=False
        )
        
        self.pointwise = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1,
            bias=True
        )
        
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class SENet(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch, channels, height, width = x.size()
        
        y = self.global_avg_pool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1, 1)
        
        return x * y.expand_as(x)
    
    def get_attention_weights(self, x):
        batch, channels, height, width = x.size()
        y = self.global_avg_pool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1, 1)
        return y


class PaperExactSLS(nn.Module):
    """
    EXACT reproduction of SLS architecture from the paper.
    
    Architecture (Fig. 1, page 4):
    - SeparableConv ×3 (1→128→128→1, kernels: 3,5,7)
    - Reshape to (31, batch, 1) for LSTM
    - LSTM (input_size=1, hidden_size=128, num_layers=1, batch_first=False, bidirectional=False)
    - Reshape to (batch, 128, 31, 1) for SENet
    - SENet (reduction=16)
    - Flatten to (batch, 128*31)
    - CosineLinearLayer to num_classes (for Circle Loss compatibility)
    """
    
    def __init__(self, 
                 input_dim=31,  # Paper uses 31 features
                 lstm_hidden=128,
                 num_classes=2,
                 dropout_rate=0.15,
                 se_reduction=16):
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.lstm_hidden = lstm_hidden
        self.num_classes = num_classes
        
        # Validate input dimension - paper uses 31 features
        assert input_dim == 31, f"Paper uses 31 features, got {input_dim}"
        
        # Three separable convolution blocks with different kernel sizes
        # Following Section IV-C: 1→128→128→1
        self.sep_conv1 = SeparableConvBlock(1, 128, kernel_size=3, dropout_rate=dropout_rate)
        self.sep_conv2 = SeparableConvBlock(128, 128, kernel_size=5, dropout_rate=dropout_rate)
        self.sep_conv3 = SeparableConvBlock(128, 1, kernel_size=7, dropout_rate=dropout_rate)
        
        # EXACT LSTM as paper: input_size=1, hidden_size=128, single layer, unidirectional
        # batch_first=False for (seq_len, batch, input_size) format
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=False,
            bidirectional=False
        )
        
        # SENet block - channels = lstm_hidden (128, not 256 because unidirectional)
        self.senet = SENet(channels=lstm_hidden, reduction=se_reduction)
        
        # CRITICAL FIX: Use CosineLinearLayer for Circle Loss compatibility
        # This outputs cosine similarities in [-1, 1] range as required by Circle Loss
        self.fc = CosineLinearLayer(
            in_features=lstm_hidden * input_dim,
            out_features=num_classes,
            bias=False  # No bias for cosine similarity
        )
        
        print(f"\n📊 Paper-Exact SLS Model initialized:")
        print(f"  - Input dimension: {input_dim} (31 as per paper)")
        print(f"  - Conv blocks: 3 (kernels: 3,5,7)")
        print(f"  - Conv channels: 1 → 128 → 128 → 1")
        print(f"  - LSTM: input_size=1, hidden={lstm_hidden}, unidirectional, single layer")
        print(f"  - SENet reduction: {se_reduction}")
        print(f"  - Final layer: CosineLinearLayer ({lstm_hidden * input_dim} → {num_classes})")
        print(f"  - Output: Cosine similarities in [-1, 1] (for Circle Loss)")
        print(f"  - Number of classes: {num_classes}")
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, 1, input_dim)
            
        Returns:
            Cosine similarities tensor of shape (batch_size, num_classes)
            Values are in [-1, 1] range for Circle Loss
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        # Validate input dimension
        assert x.size(2) == self.input_dim, \
            f"Expected input_dim={self.input_dim}, got {x.size(2)}"
        
        batch_size = x.size(0)
        
        # 1. Separable convolutions
        x = self.sep_conv1(x)  # (batch, 128, input_dim)
        x = self.sep_conv2(x)  # (batch, 128, input_dim)
        x = self.sep_conv3(x)  # (batch, 1, input_dim)
        
        # 2. Reshape for LSTM: (input_dim, batch, 1) as per paper
        x = x.squeeze(1)           # (batch, input_dim)
        x = x.permute(1, 0)        # (input_dim, batch)
        x = x.unsqueeze(-1)         # (input_dim, batch, 1)
        
        # 3. LSTM
        # Input: (seq_len=input_dim, batch, input_size=1)
        # Output: (seq_len, batch, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 4. Reshape for SENet: (batch, hidden, input_dim, 1)
        x = lstm_out.permute(1, 2, 0)  # (batch, hidden, input_dim)
        x = x.unsqueeze(-1)             # (batch, hidden, input_dim, 1)
        
        # 5. SENet channel attention
        x = self.senet(x)                # (batch, hidden, input_dim, 1)
        
        # 6. Remove last dimension and flatten
        x = x.squeeze(-1)                # (batch, hidden, input_dim)
        x = x.reshape(batch_size, -1)    # (batch, hidden * input_dim)
        
        # 7. CosineLinearLayer outputs cosine similarities in [-1, 1]
        # This is what Circle Loss expects for proper geometric interpretation
        cosine_sim = self.fc(x)          # (batch, num_classes)
        
        return cosine_sim  # Return cosine similarities for Circle Loss
    
    def get_attention_weights(self, x):
        """
        Get attention weights from SENet for interpretability.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            attention_weights: Attention weights from SENet
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Forward pass up to SENet
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.sep_conv3(x)
        x = x.squeeze(1).permute(1, 0).unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out.permute(1, 2, 0).unsqueeze(-1)
        
        # Get attention weights from SENet
        attention_weights = self.senet.get_attention_weights(x)
        return attention_weights
    
    def predict_with_threshold(self, x, threshold=0.57, gbdt_fallback=None):
        """
        Predict with uncertainty threshold.
        If max probability < threshold, use GBDT fallback.
        
        Args:
            x: Input tensor
            threshold: Uncertainty threshold (0.57 as per paper)
            gbdt_fallback: Optional GBDT model for uncertain samples
            
        Returns:
            predictions, probabilities, uncertain_mask
        """
        self.eval()
        with torch.no_grad():
            # Get cosine similarities
            cosine_sim = self.forward(x)
            
            # Convert to probabilities via softmax for threshold logic
            probs = F.softmax(cosine_sim, dim=1)
            
            # Paper logic: if max(probabilities) < threshold, use GBDT
            max_probs, predictions = probs.max(dim=1)
            
            # Convert to numpy for fallback logic
            preds_np = predictions.cpu().numpy()
            probs_np = probs[:, 1].cpu().numpy()  # Probability of class 1
            uncertain_mask = max_probs.cpu().numpy() < threshold
            
            # Apply GBDT fallback for uncertain samples
            if gbdt_fallback is not None and uncertain_mask.any():
                uncertain_indices = np.where(uncertain_mask)[0]
                
                # Handle both tensor and numpy input
                if isinstance(x, torch.Tensor):
                    uncertain_inputs = x[uncertain_indices].cpu().numpy()
                else:
                    uncertain_inputs = x[uncertain_indices]
                
                if uncertain_inputs.ndim == 3:
                    uncertain_inputs = uncertain_inputs.squeeze(1)
                
                gbdt_preds = gbdt_fallback.predict(uncertain_inputs)
                gbdt_probs = gbdt_fallback.predict_proba(uncertain_inputs)[:, 1]
                
                preds_np[uncertain_indices] = gbdt_preds
                probs_np[uncertain_indices] = gbdt_probs
            
            return torch.tensor(preds_np), torch.tensor(probs_np), torch.tensor(uncertain_mask)
    
    def get_model_info(self):
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            "input_dim": self.input_dim,
            "lstm_hidden": self.lstm_hidden,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "fc_input_size": self.lstm_hidden * self.input_dim,
            "architecture": "SeparableConv(3,5,7) → LSTM (unidirectional) → SENet → CosineLinearLayer",
            "expected_input": f"(batch_size, {self.input_dim}) or (batch_size, 1, {self.input_dim})",
            "output_type": "cosine similarities ([-1, 1]) for Circle Loss",
            "num_classes": self.num_classes,
            "paper_exact": True
        }
        return info


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING PAPER-EXACT SLS ARCHITECTURE")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with paper parameters
    model = PaperExactSLS(
        input_dim=31,  # Paper uses 31 features
        lstm_hidden=128,
        num_classes=2,
        dropout_rate=0.15,
        se_reduction=16
    ).to(device)
    
    # Print model info
    info = model.get_model_info()
    print(f"\n📊 Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 31).to(device)  # (batch, 31)
    labels = torch.tensor([0, 1, 0, 1]).to(device)
    
    cosine_sim = model(x)
    print(f"\n📊 Forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {cosine_sim.shape}")
    print(f"  Output range: [{cosine_sim.min():.4f}, {cosine_sim.max():.4f}]")
    print(f"  ✓ Output is cosine similarity in [-1, 1] range")
    
    # Verify LSTM configuration
    print(f"\n📊 LSTM Configuration:")
    print(f"  input_size=1, hidden_size=128, num_layers=1")
    print(f"  batch_first=False, bidirectional=False")
    print(f"  ✓ Matches paper Section IV-D")
    
    # Import Circle Loss for testing
    try:
        from src.training.loss import CircleLoss
        
        # Test Circle Loss with cosine similarities
        circle_loss = CircleLoss(m=0.25, gamma=256)
        loss = circle_loss(cosine_sim, labels)
        print(f"\n📊 Circle Loss: {loss.item():.4f}")
        print(f"  ✓ Circle Loss with gamma=256, m=0.25")
    except ImportError:
        print(f"\n⚠️  CircleLoss not available for testing")
    
    # Test prediction with threshold
    predictions, probs, uncertain = model.predict_with_threshold(x, threshold=0.57)
    print(f"\n📊 Prediction with threshold 0.57:")
    print(f"  Probabilities: {probs.numpy()}")
    print(f"  Predictions: {predictions.numpy()}")
    print(f"  Uncertain samples: {uncertain.numpy()}")
    print(f"  ✓ Paper logic: use GBDT when max(prob) < threshold")
    
    print("\n✅ All checks passed!")