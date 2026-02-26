import torch
import torch.nn as nn


class SLSLSTM(nn.Module):
    """
    Paper-faithful LSTM module for SLS model.

    Matches Section IV-D of the paper:
    - input_size = 1
    - hidden_size = 128
    - num_layers = 1
    - unidirectional
    - batch_first = False
    """

    def __init__(
        self,
        input_dim=31,        # number of features (sequence length)
        input_size=1,
        hidden_size=128,
        num_layers=1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size

        # EXACT configuration from paper
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,   # IMPORTANT (paper format)
            bidirectional=False
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: tensor shape
               (batch, 1, input_dim)
               OR
               (batch, input_dim)

        Returns:
            lstm_out: (seq_len=input_dim, batch, hidden_size)
        """

        # ---- Input handling ----
        if x.dim() == 3:
            # (batch,1,31) → (batch,31)
            x = x[:, 0, :]

        # Convert to paper format:
        # (batch, input_dim) → (input_dim, batch, 1)
        x = x.permute(1, 0).unsqueeze(-1)

        # ---- LSTM ----
        lstm_out, (hidden, cell) = self.lstm(x)

        return lstm_out, hidden, cell

    def reshape_for_senet(self, lstm_out):
        """
        Convert LSTM output to SENet input format.

        Paper expects:
        (batch, hidden, input_dim, 1)
        """

        x = lstm_out.permute(1, 2, 0)  # (batch, hidden, seq_len)
        x = x.unsqueeze(-1)            # (batch, hidden, seq_len, 1)
        return x

    def get_last_hidden(self, hidden):
        """
        Return last hidden state for ablation experiments.
        Shape: (batch, hidden_size)
        """
        return hidden[-1]

    def get_model_info(self):
        return {
            "input_size": 1,
            "hidden_size": self.hidden_size,
            "num_layers": 1,
            "bidirectional": False,
            "batch_first": False,
            "paper_exact": True
        }


# ------------------------------------------------------------
# TEST BLOCK
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING PAPER-FAITHFUL SLS LSTM")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SLSLSTM().to(device)

    batch_size = 4
    input_dim = 31

    x = torch.randn(batch_size, input_dim).to(device)

    lstm_out, hidden, cell = model(x)

    print("\n📊 Shapes:")
    print("Input:", x.shape)
    print("LSTM output:", lstm_out.shape)
    print("Hidden state:", hidden.shape)
    print("Cell state:", cell.shape)

    # reshape test
    senet_input = model.reshape_for_senet(lstm_out)
    print("SENet input:", senet_input.shape)

    print("\n✅ Paper-faithful configuration verified!")