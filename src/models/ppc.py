import torch
import torch.nn as nn
import torch.nn.functional as F


class PPC(nn.Module):
    def __init__(
        self,
        input_dim=8,     
        gru_hidden=32,
        cnn_filters=32,
        kernel_size=3,
        num_classes=2
    ):
        super().__init__()

        # -------- GRU (GLOBAL) --------
        self.gru = nn.GRU(
            input_dim,
            gru_hidden,
            batch_first=True
        )

        self.conv = nn.Conv1d(
            input_dim,
            cnn_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        # -------- FINAL --------
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden + cnn_filters, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F)

        # -------- GRU branch --------
        gru_out, _ = self.gru(x)   # (B, T, H)
        sR = torch.mean(gru_out, dim=1)  # mean pooling

        # -------- CNN branch --------
        x_cnn = x.transpose(1, 2)  # (B, F, T)
        conv_out = F.relu(self.conv(x_cnn))  # (B, C, T)
        sC = torch.mean(conv_out, dim=2)  # mean pooling

        # -------- CONCAT --------
        s = torch.cat([sR, sC], dim=1)

        return self.fc(s)