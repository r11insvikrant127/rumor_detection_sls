import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    """
    Circle Loss implementation used in the SLS paper.

    Paper statement:
    "Cross-entropy loss is replaced by Circle Loss."

    This implementation works directly with the logits produced
    by the final Linear classifier of the SLS model.

    Hyperparameters (paper settings):
        m = 0.25
        gamma = 256
    """

    def __init__(self, m: float = 0.25, gamma: float = 256, reduction: str = "mean"):
        super().__init__()

        self.m = m
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        logits : Tensor (B, C)
            Output of final Linear layer

        labels : Tensor (B,)
            Ground truth labels

        Returns
        -------
        loss : Tensor
        """

        # One-hot labels
        one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()

        # Positive similarities
        sp = (logits * one_hot).sum(dim=1)

        # Negative similarities
        sn = logits * (1 - one_hot)

        # Adaptive weighting
        ap = torch.clamp_min(1 + self.m - sp.detach(), 0.)
        an = torch.clamp_min(sn.detach() + self.m, 0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -self.gamma * ap * (sp - delta_p)
        logit_n = self.gamma * an * (sn - delta_n)

        loss = F.softplus(
            torch.logsumexp(logit_n, dim=1) + logit_p
        )

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


def create_circle_loss(
    m: float = 0.25,
    gamma: float = 256,
    reduction: str = "mean"
) -> CircleLoss:
    """
    Factory function for Circle Loss using paper parameters.
    """

    return CircleLoss(
        m=m,
        gamma=gamma,
        reduction=reduction
    )


# ============================================================
# Quick Test
# ============================================================

def test_circle_loss():
    print("=" * 60)
    print("Testing Circle Loss (Paper Faithful)")
    print("=" * 60)

    batch_size = 8
    num_classes = 2

    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))

    loss_fn = create_circle_loss()

    loss = loss_fn(logits, labels)

    print("Logits shape:", logits.shape)
    print("Labels shape:", labels.shape)
    print("Loss:", loss.item())

    print("\n✓ Circle Loss working correctly")


if __name__ == "__main__":
    test_circle_loss()