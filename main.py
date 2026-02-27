"""
Main entry point for PAPER-FAITHFUL SLS Rumor Detection System
(31 kernel-subtree features only)

Implements pipeline described in:
"A Novel and High-Accuracy Rumor Detection Approach using
Kernel Subtree and Deep Learning Networks"
"""

"""
✅ REAL PURPOSE OF THIS FILE (In YOUR case)

After training, main.py is used for 3 things only:

① Prediction / Evaluation (MOST IMPORTANT)

This is the main usage.

You now have a trained SLS model.

To classify new events:

python main.py predict \
    --model experiments/best_model \
    --data test.csv
What happens internally:
Load trained SLS model
        ↓
Load CSV features
        ↓
Run forward pass
        ↓
Compute softmax probabilities
        ↓
Apply threshold (0.57)
        ↓
If uncertain → GBDT prediction
        ↓
Save predictions.csv

This corresponds EXACTLY to paper Section IV-F:

GBDT assists model when prediction confidence is low 

7bda6a77-adbe-4e6a-aefa-783ef01…

👉 THIS step is where the hybrid SLS+GBDT system actually exists.

Training alone ≠ final model from paper.

② Running Ablation Study (Paper Experiments)

Paper compares:

pSLS → propagation only

uSLS → user only

cSLS → content only

You run:

python main.py ablation --data train.csv

This reproduces Fig.4 in the paper.

You normally do this when writing report/thesis.

③ System Testing / Debugging
python main.py test

Checks:

model forward pass

feature dimension

architecture correctness

Useful if something crashes.
"""

import argparse
import sys
from pathlib import Path
import json
import traceback
import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Paper-Faithful SLS Rumor Detection System"
    )

    subparsers = parser.add_subparsers(dest="command")

    # ---------------- TRAIN ----------------
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", default="configs/default.yaml")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--experiment", default=None)
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--batch_size", type=int)
    train_parser.add_argument("--seed", type=int)

    # ---------------- PREDICT ----------------
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--data", required=True)
    predict_parser.add_argument("--output", default="predictions.csv")

    # PAPER threshold (Section IV-F)
    predict_parser.add_argument(
        "--threshold",
        type=float,
        default=0.57,
        help="Paper threshold (default = 0.57)"
    )

    predict_parser.add_argument("--batch_size", type=int, default=32)

    # ---------------- FEATURE EXTRACTION ----------------
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--input", required=True)
    extract_parser.add_argument("--output", required=True)
    extract_parser.add_argument("--batch_size", type=int, default=100)

    # ---------------- ABLATION ----------------
    ablation_parser = subparsers.add_parser("ablation")
    ablation_parser.add_argument("--config", default="configs/default.yaml")
    ablation_parser.add_argument("--data", required=True)
    ablation_parser.add_argument("--output_dir", default="experiments/ablation")
    ablation_parser.add_argument("--seed", type=int, default=42)

    # Only paper feature groups
    ablation_parser.add_argument(
        "--feature_groups",
        nargs="+",
        default=["propagation", "user", "content"]
    )

    # ---------------- TEST ----------------
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


# ============================================================
# FEATURE EXTRACTION (31 FEATURES ONLY)
# ============================================================

def extract_features(args):

    from src.preprocessing.feature_extractor import FeatureExtractor

    print("\nExtracting PAPER features (31 kernel-subtree features)")

    with open(args.input, "r") as f:
        raw_data = json.load(f)

    extractor = FeatureExtractor()
    all_feature_names = extractor.get_feature_names()

    # PAPER: ONLY FIRST 31 FEATURES
    feature_names = all_feature_names[:31]

    features = []
    event_ids = []

    items = list(raw_data.items())

    for i in tqdm(range(0, len(items), args.batch_size)):

        batch = items[i:i + args.batch_size]

        for event_id, event_data in batch:
            try:
                vec = extractor.extract_features(event_data)
                vec = vec[:31]   # HARD LOCK TO PAPER FEATURES

                features.append(vec)
                event_ids.append(event_id)

            except Exception as e:
                print(f"⚠ Failed event {event_id}: {e}")

    df = pd.DataFrame(features, columns=feature_names)
    df.insert(0, "event_id", event_ids)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✓ Saved {len(df)} events with 31 features → {args.output}")


# ============================================================
# TEST UTILITIES
# ============================================================

def run_tests(verbose=False):

    print("=" * 60)
    print("TESTING PAPER SLS SYSTEM")
    print("=" * 60)

    from src.models.sls import SLSModel
    import torch

    input_dim = 31

    model = SLSModel(input_dim=input_dim, num_classes=2)

    print(f"✓ Model created (input_dim={input_dim})")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    test_input = torch.randn(2, 1, input_dim)
    output = model(test_input)

    print(f"✓ Forward pass OK: {output.shape}")

    print("\n✓ All tests passed.")


# ============================================================
# MAIN ROUTER
# ============================================================

def main():

    args = parse_args()

    if args.command == "train":

        from src.scripts.train import train

        kwargs = {
            "config_path": args.config,
            "data_path": args.data,
            "experiment_name": args.experiment,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
        }

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        train(**kwargs)

    elif args.command == "predict":

        from src.scripts.predict import predict

        predict(
            model_path=args.model,
            data_path=args.data,
            output_path=args.output,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )

    elif args.command == "extract":
        extract_features(args)

    elif args.command == "ablation":

        from src.scripts.ablation_study import run_ablation_study

        run_ablation_study(
            config_path=args.config,
            data_path=args.data,
            output_dir=args.output_dir,
            seed=args.seed,
            feature_groups=args.feature_groups,
        )

    elif args.command == "test":
        run_tests(args.verbose)

    else:
        print("\nAvailable commands:")
        print("  train")
        print("  predict")
        print("  extract")
        print("  ablation")
        print("  test")
        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)