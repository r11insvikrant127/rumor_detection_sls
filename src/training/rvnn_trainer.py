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

        for epoch in range(self.epochs):

            total_loss = 0
            self.model.train()

            for root, label in zip(roots_train, labels_train):

                # ---- Assign TF-IDF ----
                assign_tfidf_to_nodes(root)

                # ---- Convert to RvNN input ----
                data = build_rvnn_inputs(root)

                X_word = data["X_word"]
                X_index = data["X_index"]
                tree = data["tree"]

                y = torch.tensor([label], dtype=torch.long, device=self.device)

                out = self.model(X_word, X_index, tree)

                loss = self.criterion(out, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"[RvNN] Epoch {epoch+1} Loss: {total_loss / len(roots_train):.4f}")

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