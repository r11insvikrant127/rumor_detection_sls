"""
Check if all required dependencies are installed.
"""
import importlib
import sys

def check_dependency(module_name, pip_name=None):
    """Check if a module is installed."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError:
        pip_name = pip_name or module_name
        print(f"❌ {module_name} - Install: pip install {pip_name}")
        return False

print("=" * 60)
print("DEPENDENCY CHECK")
print("=" * 60)

dependencies = [
    ("torch", "torch"),
    ("sklearn", "scikit-learn"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("networkx", "networkx"),
    ("textblob", "textblob"),
    ("spacy", "spacy"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("yaml", "pyyaml"),
    ("tqdm", "tqdm"),
    ("joblib", "joblib"),
    ("wandb", "wandb"),           
    ("hydra", "hydra-core"),
]

all_ok = True
for module, pip_name in dependencies:
    if not check_dependency(module, pip_name):
        all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("✅ All dependencies installed!")
    print("\nYou can now run: python src/scripts/train.py")
else:
    print("⚠ Missing dependencies. Install with:")
    print("\npip install torch scikit-learn pandas numpy networkx textblob spacy matplotlib seaborn pyyaml tqdm joblib")