"""
PAPER-ALIGNED Prediction Pipeline for SLS Rumor Detection.

Implements inference exactly as described in the paper:
1. Extract 31 features
2. Apply training normalization
3. Predict using SLS
4. If max(prob) < 0.57 → use GBDT
"""

import torch
import numpy as np
import json
from pathlib import Path
from src.utils.config import load_config

from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.feature_normalizer import FeatureNormalizer
from src.models.sls import PaperExactSLS
from src.models.gbdt_wrapper import GBDTWrapper
from src.utils.helpers import get_device


# ------------------------------------------------------------
# Predictor
# ------------------------------------------------------------

class RumorPredictor:
    """Paper-faithful SLS + GBDT prediction pipeline."""

    PAPER_FEATURE_DIM = 31
    

    def __init__(self, model_dir):

        self.device = get_device()
        self.model_dir = Path(model_dir)

        print(f"🔍 Loading models from {self.model_dir}")

        # --------------------------------------------------
        # Feature extractor
        # --------------------------------------------------
        self.feature_extractor = FeatureExtractor()
        self.feature_names = self.feature_extractor.get_feature_names()[:31]

        assert len(self.feature_names) == self.PAPER_FEATURE_DIM, \
            "Paper requires exactly 31 features."

        # --------------------------------------------------
        # Load normalizer (trained stats)
        # --------------------------------------------------
        normalizer_path = self.model_dir / "feature_normalizer.pkl"

        if not normalizer_path.exists():
            raise FileNotFoundError("feature_normalizer.pkl not found")

        self.normalizer = FeatureNormalizer()
        self.normalizer.load(str(normalizer_path))

        stats = self.normalizer.get_stats()
        assert stats["n_features"] == self.PAPER_FEATURE_DIM, \
            "Normalizer feature mismatch."

        print("✅ Feature normalizer loaded")

        # --------------------------------------------------
        # Load SLS model
        # --------------------------------------------------
        model_path = self.model_dir / "sls_model.pth"
        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = PaperExactSLS(
            input_dim=31,
            lstm_hidden=128,
            num_classes=2
        )

        state_dict = checkpoint.get(
            "model_state_dict", checkpoint
        )
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        print("✅ SLS model loaded")

        # --------------------------------------------------
        # Load GBDT (optional but paper uses it)
        # --------------------------------------------------
        gbdt_path = self.model_dir / "gbdt_model.joblib"

        if gbdt_path.exists():
            self.gbdt = GBDTWrapper()
            self.gbdt.load(str(gbdt_path))
            print("✅ GBDT loaded")
        else:
            self.gbdt = None
            print("⚠️ GBDT not found — SLS only mode")

        config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

        self.config = load_config(str(config_path))
        self.threshold = self.config.gbdt.threshold

        print(f"📊 Threshold = {self.threshold}")
        print("🎯 Predictor ready")

    # ------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------

    def extract_features(self, event):

        features = self.feature_extractor.extract_features(event)[:31]

        if len(features) != self.PAPER_FEATURE_DIM:
            raise ValueError(
                f"Expected 31 features, got {len(features)}"
            )

        features = np.nan_to_num(features).reshape(1, -1)

        return features

    # ------------------------------------------------------------
    # Hybrid Decision (Paper Logic)
    # ------------------------------------------------------------

    def hybrid_decision(self, sls_probs, features_raw, features_norm):

        sls_pred = np.argmax(sls_probs)
        max_prob = np.max(sls_probs)

        # Paper rule
        if max_prob < self.threshold and self.gbdt is not None:

            gbdt_pred = self.gbdt.predict(features_raw)[0]
            gbdt_probs = self.gbdt.predict_proba(features_raw)[0]

            return {
                "prediction": int(gbdt_pred),
                "confidence": float(np.max(gbdt_probs)),
                "method": "gbdt",
                "is_uncertain": True,
                "probabilities": {
                    "non-rumor": float(gbdt_probs[0]),
                    "rumor": float(gbdt_probs[1]),
                },
            }

        else:
            return {
                "prediction": int(sls_pred),
                "confidence": float(max_prob),
                "method": "sls",
                "is_uncertain": False,
                "probabilities": {
                    "non-rumor": float(sls_probs[0]),
                    "rumor": float(sls_probs[1]),
                },
            }

    # ------------------------------------------------------------
    # Single Prediction
    # ------------------------------------------------------------

    def predict_event(self, event):

        # Feature extraction
        features = self.extract_features(event)

        # Normalize
        features_norm = self.normalizer.transform(features)

        # SLS inference
        with torch.no_grad():
            tensor = torch.tensor(features_norm, dtype=torch.float32)\
                .unsqueeze(1)\
                .to(self.device)

            outputs = self.model(tensor)
            temperature = getattr(self.config, "temperature", 1.0) #you used temperature in predict() in trainer.py
            probs = torch.softmax(outputs / temperature, dim=1)

        sls_probs = probs[0].cpu().numpy()

        result = self.hybrid_decision(sls_probs, features, features_norm)

        result["prediction"] = (
            "rumor" if result["prediction"] == 1 else "non-rumor"
        )

        return result


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def load_event(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # label not needed for inference
    data.pop("label", None)
    return data


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--event-json", required=True)

    args = parser.parse_args()

    predictor = RumorPredictor(args.model_dir)

    event = load_event(args.event_json)
    result = predictor.predict_event(event)

    print("\n📊 Prediction")
    print(result)


if __name__ == "__main__":
    main()