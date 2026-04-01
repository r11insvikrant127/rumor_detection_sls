import torch
import torch.nn as nn
import numpy as np
from src.preprocessing import (
    fit_tfidf,
    assign_tfidf_to_nodes,
    build_rvnn_inputs
)


class RvNNTrainer:

    def __init__(self, model, device="cuda", lr=0.005, epochs=50):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs

        # ✅ KEEP CrossEntropy (better than paper MSE for classification)
        self.criterion = nn.CrossEntropyLoss()

        # ✅ KEEP Adam (more stable than paper SGD)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    # =====================================================
    # TRAIN
    # =====================================================
    def train(self, roots_train, labels_train):

        print("🔧 Fitting TF-IDF...")
        fit_tfidf(roots_train)

        best_loss = float("inf")
        patience = 5
        no_improve = 0
        best_state = None

        for epoch in range(self.epochs):

            total_loss = 0
            self.model.train()

            for root, label in zip(roots_train, labels_train):

                assign_tfidf_to_nodes(root)
                data = build_rvnn_inputs(root)

                X_word = data["X_word"]
                X_index = data["X_index"]
                tree = data["tree"]

                y = torch.tensor([label], dtype=torch.long).to(self.device)

                out = self.model(X_word, X_index, tree)
                loss = self.criterion(out, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            epoch_loss = total_loss / len(roots_train)

            print(f"[RvNN] Epoch {epoch+1} Loss: {epoch_loss:.4f}")

            # 🔴 EARLY STOPPING
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = self.model.state_dict()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"[RvNN] Early stopping at epoch {epoch+1}")
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"[RvNN] Best Loss: {best_loss:.4f}")

    # =====================================================
    # PREDICT
    # =====================================================
    def predict(self, roots):

        self.model.eval()
        preds = []

        with torch.no_grad():
            for root in roots:

                assign_tfidf_to_nodes(root)
                data = build_rvnn_inputs(root)

                out = self.model(
                    data["X_word"],
                    data["X_index"],
                    data["tree"]
                )

                preds.append(torch.argmax(out, dim=1).item())

        return np.array(preds)