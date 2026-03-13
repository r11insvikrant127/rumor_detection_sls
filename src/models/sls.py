import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Separable Convolution Block (Section IV-C)
# ============================================================
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate=0.15):
        super().__init__()

        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False,
        )

        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


# ============================================================
# SENet Block (Section IV-E)
# ============================================================
class SENet(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ============================================================
# PAPER-FAITHFUL SLS MODEL
# ============================================================
class PaperExactSLS(nn.Module):
    """
    Paper exact architecture from:

    "A Novel and High-Accuracy Rumor Detection Approach
    using Kernel Subtree and Deep Learning Networks"

    Pipeline:
        SeparableConv ×3
        → LSTM
        → SENet
        → FC (Linear classifier)
    """

    def __init__(
        self,
        input_dim=31,
        lstm_hidden=128,
        num_classes=2,
        dropout_rate=0.15,
        se_reduction=16,
    ):
        super().__init__()

        # Allow ablation studies with different feature counts
        assert input_dim > 0, "Input dimension must be positive."

        self.input_dim = input_dim
        self.lstm_hidden = lstm_hidden
        self.num_classes = num_classes

        # --------------------------------------------------
        # Separable Convolutions (1 → 128 → 128 → 1)
        # --------------------------------------------------
        self.sep1 = SeparableConvBlock(1, 128, 3, dropout_rate)
        self.sep2 = SeparableConvBlock(128, 128, 5, dropout_rate)
        self.sep3 = SeparableConvBlock(128, 1, 7, dropout_rate)

        # --------------------------------------------------
        # LSTM (Section IV-D)
        # --------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=False,
            batch_first=False,
        )

        # --------------------------------------------------
        # SENet
        # --------------------------------------------------
        self.senet = SENet(lstm_hidden, se_reduction)

        # --------------------------------------------------
        # FINAL CLASSIFIER (Paper uses FC + Softmax)
        # --------------------------------------------------
        self.fc = nn.Linear(
            in_features=lstm_hidden * input_dim,
            out_features=num_classes
        )

    # ========================================================
    # Forward Pass
    # ========================================================
    def forward(self, x):
        """
        Input:
            (batch, 31) or (batch, 1, 31)

        Output:
            logits (batch, num_classes)
        """

        if x.dim() == 2:
            x = x.unsqueeze(1)

        assert x.size(2) == self.input_dim

        batch_size = x.size(0)

        # --- Separable Convolutions ---
        x = self.sep1(x)
        x = self.sep2(x)
        x = self.sep3(x)

        # --- reshape for LSTM ---
        x = x.squeeze(1)        # (B, L)
        x = x.permute(1, 0)     # (L, B)
        x = x.unsqueeze(-1)     # (L, B, 1)

        # --- LSTM ---
        lstm_out, _ = self.lstm(x)

        # --- reshape for SENet ---
        x = lstm_out.permute(1, 2, 0).unsqueeze(-1)

        # --- SENet ---
        x = self.senet(x)

        # --- Flatten ---
        x = x.squeeze(-1).reshape(batch_size, -1)

        # --- Linear Classifier ---
        logits = self.fc(x)

        return logits

    # ========================================================
    # Probability helper
    # ========================================================
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs

    # ========================================================
    # Prediction helper
    # ========================================================
    def predict(self, x):
        probs = self.predict_proba(x)
        return probs.argmax(dim=1)

    # ========================================================
    # Model info
    # ========================================================
    def get_model_info(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "architecture":
                "SeparableConv(3,5,7) → LSTM → SENet → Linear → Softmax",
            "input_dim": self.input_dim,
            "lstm_hidden": self.lstm_hidden,
            "params_total": total,
            "params_trainable": trainable,
            "paper_exact": True,
        }