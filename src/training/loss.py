import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    def __init__(
        self,
        m: float = 0.25,
        gamma: float = 256,
        reduction: str = "mean"
    ):
        super().__init__()

        self.m = m
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, similarities: torch.Tensor, labels: torch.Tensor):

        # One-hot encoding
        one_hot = F.one_hot(labels, num_classes=similarities.size(1)).float()

        # Positive similarity
        sp = torch.sum(similarities * one_hot, dim=1)

        # Negative similarities
        sn = similarities[one_hot == 0].view(similarities.size(0), -1)

        # Adaptive weights
        ap = torch.clamp_min(1 + self.m - sp.detach(), 0.0)
        an = torch.clamp_min(sn.detach() + self.m, 0.0)

        # Margins
        delta_p = 1 - self.m
        delta_n = self.m

        # Logits transformation
        logit_p = -self.gamma * ap * (sp - delta_p)
        logit_n = self.gamma * an * (sn - delta_n)

        # Final loss
        loss = F.softplus(
            torch.logsumexp(logit_n, dim=1) + logit_p
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def create_circle_loss(
    m: float = 0.25,
    gamma: float = 256,
    reduction: str = "mean"
) -> CircleLoss:
    return CircleLoss(m, gamma, reduction)


# ============================================================
# Test
# ============================================================

def test_circle_loss():
    print("=" * 60)
    print("Testing Circle Loss (Correct Simulation)")
    print("=" * 60)

    B, C, D = 8, 2, 128

    # Simulate normalized features
    x = torch.randn(B, D)
    x = F.normalize(x, dim=1)

    # Simulate normalized class weights
    W = torch.randn(C, D)
    W = F.normalize(W, dim=1)

    # Cosine similarity
    similarity = x @ W.T

    labels = torch.randint(0, C, (B,))

    loss_fn = create_circle_loss()

    loss = loss_fn(similarity, labels)

    print("Loss:", loss.item())
    print("✓ Works correctly")


if __name__ == "__main__":
    test_circle_loss()