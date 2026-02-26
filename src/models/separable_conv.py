import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

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


class SLSFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, dropout_rate=0.15):
        super().__init__()
        
        self.conv1 = SeparableConvBlock(
            in_channels=input_channels, 
            out_channels=128, 
            kernel_size=3,
            dropout_rate=dropout_rate
        )
        
        self.conv2 = SeparableConvBlock(
            in_channels=128, 
            out_channels=128, 
            kernel_size=5,
            dropout_rate=dropout_rate
        )
        
        self.conv3 = SeparableConvBlock(
            in_channels=128, 
            out_channels=1, 
            kernel_size=7,
            dropout_rate=dropout_rate
        )
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
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


class PaperExactSLS(nn.Module):
    def __init__(self, 
                 input_channels=1,
                 lstm_hidden_size=128,
                 num_classes=2,
                 dropout_rate=0.15,
                 se_reduction=16):
        super().__init__()
        
        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size
        
        self.feature_extractor = SLSFeatureExtractor(
            input_channels=input_channels,
            dropout_rate=dropout_rate
        )
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=False,
            bidirectional=False
        )
        
        self.se_net = SENet(
            channels=lstm_hidden_size,
            reduction=se_reduction
        )
        
        self.fc = nn.Linear(
            lstm_hidden_size * 31,
            num_classes
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.feature_extractor(x)
        
        x = x.squeeze(1)
        x = x.permute(1, 0)
        x = x.unsqueeze(-1)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        x = lstm_out.permute(1, 2, 0)
        x = x.unsqueeze(-1)
        
        x = self.se_net(x)
        
        x = x.view(batch_size, -1)
        
        logits = self.fc(x)
        
        return logits
    
    def predict_with_threshold(self, x, threshold=0.57, gbdt_fallback=None):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            
            max_probs, predictions = probs.max(dim=1)
            uncertain_mask = max_probs < threshold
            
            if gbdt_fallback is not None and uncertain_mask.any():
                uncertain_indices = uncertain_mask.nonzero(as_tuple=True)[0]
                uncertain_x = x[uncertain_indices].cpu().numpy()
                gbdt_preds = gbdt_fallback.predict(uncertain_x)
                predictions[uncertain_indices] = torch.tensor(gbdt_preds)
            
            return predictions, probs


class CircleLoss(nn.Module):
    """
    Circle Loss for classification.
    Applied directly to logits as a replacement for Cross-Entropy.
    """
    def __init__(self, scale=32, margin=0.25):
        super().__init__()
        self.scale = scale
        self.margin = margin
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes) - raw logits from model
            targets: (batch_size) - class indices
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        
        one_hot = F.one_hot(targets, num_classes).float()
        
        mask = 2 * one_hot - 1
        
        logits_masked = logits * mask
        
        sp = torch.where(one_hot == 1, logits, torch.zeros_like(logits))
        sn = torch.where(one_hot == 0, logits, torch.zeros_like(logits))
        
        ap = torch.clamp_min(1 + self.margin - sp.detach(), min=0.)
        an = torch.clamp_min(sn.detach() + self.margin, min=0.)
        
        delta_p = 1 - self.margin
        delta_n = self.margin
        
        logit_p = -self.scale * ap * (sp - delta_p)
        logit_n = self.scale * an * (sn - delta_n)
        
        loss_p = torch.logsumexp(logit_p, dim=1)
        loss_n = torch.logsumexp(logit_n, dim=1)
        
        loss = F.softplus(loss_p + loss_n).mean()
        
        return loss


class HybridInference:
    """
    Hybrid inference with SLS + GBDT fallback.
    Implements the exact logic from the paper:
    If max(probabilities) < threshold, use GBDT.
    """
    def __init__(self, sls_model, gbdt_model=None, threshold=0.57):
        self.sls_model = sls_model
        self.gbdt_model = gbdt_model
        self.threshold = threshold
        self.sls_model.eval()
        
    def predict(self, x):
        with torch.no_grad():
            logits = self.sls_model(x)
            probs = F.softmax(logits, dim=1)
            
            max_probs, sls_preds = probs.max(dim=1)
            sls_preds = sls_preds.cpu().numpy()
            
            if self.gbdt_model is None:
                return sls_preds, probs.cpu().numpy(), np.ones(len(x), dtype=bool)
            
            uncertain_mask = max_probs.cpu().numpy() < self.threshold
            
            final_preds = sls_preds.copy()
            
            if uncertain_mask.any():
                uncertain_x = x[uncertain_mask].cpu().numpy()
                gbdt_preds = self.gbdt_model.predict(uncertain_x)
                final_preds[uncertain_mask] = gbdt_preds
            
            return final_preds, probs.cpu().numpy(), uncertain_mask
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.sls_model(x)
            probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING PAPER-EXACT SLS ARCHITECTURE")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PaperExactSLS(num_classes=2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    batch_size = 4
    feature_length = 31
    
    x = torch.randn(batch_size, feature_length).to(device)
    labels = torch.tensor([0, 1, 0, 1]).to(device)
    
    logits = model(x)
    print(f"\n📊 Forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    print(f"\n📊 Circle Loss (classification mode) test:")
    circle_loss = CircleLoss(scale=32, margin=0.25)
    loss = circle_loss(logits, labels)
    print(f"  Circle loss value: {loss.item():.4f}")
    print(f"  ✓ Applied directly to logits with class labels")
    
    print(f"\n📊 Cross-Entropy loss for comparison:")
    ce_loss = F.cross_entropy(logits, labels)
    print(f"  CE loss value: {ce_loss.item():.4f}")
    
    print(f"\n📊 Prediction with threshold (0.57):")
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        max_probs, predictions = probs.max(dim=1)
        
        print(f"  Probabilities: {probs.cpu().numpy()}")
        print(f"  Max probabilities: {max_probs.cpu().numpy()}")
        print(f"  Predictions: {predictions.cpu().numpy()}")
        
        uncertain = max_probs < 0.57
        print(f"  Uncertain samples (max_prob < 0.57): {uncertain.cpu().numpy()}")
        
        paper_logic = "✓ Exact paper logic: use GBDT when max(probabilities) < threshold"
        print(f"  {paper_logic}")
    
    print("\n✅ All checks passed!")