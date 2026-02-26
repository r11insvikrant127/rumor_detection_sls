"""
PAPER-EXACT ablation study for SLS reproduction.
Strictly follows Section V-D, Fig. 4 & Fig. 5 from the paper.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import torch
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.feature_normalizer import FeatureNormalizer
from src.models.sls import PaperExactSLS
from src.models.gbdt_wrapper import GBDTWrapper
from src.training.trainer import SLSTrainer
from src.training.evaluator import Evaluator
from src.utils.config import ConfigManager
from src.utils.helpers import create_data_loader, set_seed

class PaperExactAblation:
    """
    STRICT PAPER-EXACT ablation study for SLS.
    
    Follows Table I (page 4) exactly for feature grouping:
    - Propagation features (6 features): indices 1-6 → 0-5 in 0-based
    - User features (13 features): indices 7-19 → 6-18 in 0-based
    - Content features (12 features): indices 20-31 → 19-30 in 0-based
    
    Reproduces:
    - Fig. 4: pSLS, uSLS, cSLS comparison
    - Fig. 5: Threshold analysis across range 0.52-0.65
    """
    
    def __init__(self, config_path):
        self.config = ConfigManager(config_path)
        set_seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = FeatureExtractor()
        
        # Get ALL features first (for reference)
        all_features = self.feature_extractor.get_feature_names()
        
        # PAPER EXACT: Map Table I indices to 0-based indexing
        # Table I (page 4):
        # Propagation: features 1-6  → indices 0-5
        # User: features 7-19        → indices 6-18
        # Content: features 20-31    → indices 19-30
        
        self.feature_names = all_features[:31]  # First 31 features
        assert len(self.feature_names) == 31, f"Paper uses 31 features, got {len(self.feature_names)}"
        
        print("=" * 80)
        print("PAPER-EXACT SLS ABLATION STUDY")
        print("=" * 80)
        print("Following Table I (page 4) feature grouping:")
        
        # Define paper-exact feature groups
        self.feature_groups = {
            'propagation': list(range(0, 6)),    # 6 features (1-6 in paper)
            'user': list(range(6, 19)),          # 13 features (7-19 in paper)
            'content': list(range(19, 31)),       # 12 features (20-31 in paper)
        }
        
        # Verify counts
        assert len(self.feature_groups['propagation']) == 6, f"Propagation should have 6 features"
        assert len(self.feature_groups['user']) == 13, f"User should have 13 features"
        assert len(self.feature_groups['content']) == 12, f"Content should have 12 features"
        
        print(f"\n📊 PAPER-EXACT Feature Groups (31 features):")
        print(f"  pSLS (propagation): {len(self.feature_groups['propagation'])} features (1-6)")
        print(f"  uSLS (user):        {len(self.feature_groups['user'])} features (7-19)")
        print(f"  cSLS (content):     {len(self.feature_groups['content'])} features (20-31)")
        
        # Show first few features in each group for verification
        for group, indices in self.feature_groups.items():
            features = [self.feature_names[i] for i in indices[:3]]
            print(f"    {group:12s}: {features}...")
        
        # For storing results
        self.results = {}
        self.cv_splits = None
        
    def load_pheme_data(self, data_dir):
        """Load PHEME data (same as original paper)."""
        print(f"\n📂 Loading PHEME data from: {data_dir}")
        
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        events = []
        
        for event_dir in data_dir.iterdir():
            if event_dir.is_dir():
                for thread_file in event_dir.glob("*.json"):
                    try:
                        with open(thread_file, 'r', encoding='utf-8') as f:
                            thread_data = json.load(f)
                            
                            formatted_event = {
                                'tweets': [],
                                'label': thread_data.get('label', 0),
                                'thread_id': thread_data.get('thread_id', '')
                            }
                            
                            # Add source tweet
                            if thread_data.get('source_tweet'):
                                source = thread_data['source_tweet']
                                formatted_tweet = {
                                    'id': source.get('id_str', ''),
                                    'text': source.get('text', ''),
                                    'user': source.get('user', {}),
                                    'created_at': source.get('created_at', ''),
                                    'response_to': None
                                }
                                formatted_event['tweets'].append(formatted_tweet)
                            
                            # Add response tweets
                            for resp in thread_data.get('response_tweets', []):
                                formatted_tweet = {
                                    'id': resp.get('id_str', ''),
                                    'text': resp.get('text', ''),
                                    'user': resp.get('user', {}),
                                    'created_at': resp.get('created_at', ''),
                                    'response_to': resp.get('in_reply_to_status_id_str')
                                }
                                formatted_event['tweets'].append(formatted_tweet)
                            
                            # Convert string label to int
                            if isinstance(formatted_event['label'], str):
                                formatted_event['label'] = 1 if formatted_event['label'].lower() in ['true', '1', 'yes', 'rumour'] else 0
                            
                            events.append(formatted_event)
                            
                    except Exception:
                        continue
        
        if not events:
            raise ValueError(f"No valid events found in {data_dir}")
        
        print(f"✅ Loaded {len(events)} events")
        return events
    
    def extract_features_from_events(self, events):
        """Extract 31 features (paper-exact)."""
        print(f"\n🔧 Extracting 31 features from {len(events)} events...")
        
        features = []
        labels = []
        
        for event in events:
            try:
                # Extract all features
                all_feats = self.feature_extractor.extract_features(event)
                
                # Take first 31 features (paper-exact)
                if len(all_feats) >= 31:
                    feat = all_feats[:31]
                else:
                    continue
                
                features.append(feat)
                labels.append(event['label'])
                
            except Exception:
                continue
        
        if not features:
            raise ValueError("No features extracted!")
        
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels)
        
        # Basic cleaning (same as paper)
        features = np.nan_to_num(features, nan=0.0)
        
        print(f"✅ Features shape: {features.shape} (31 features)")
        print(f"   Rumors: {np.sum(labels)} ({np.sum(labels)/len(labels):.1%})")
        
        return features, labels
    
    def setup_cross_validation(self, features, labels, n_folds=5):
        """Create fixed CV splits (as per paper)."""
        print(f"\n🔧 Creating {n_folds}-fold CV splits...")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        self.cv_splits = list(skf.split(features, labels))
        
        print(f"✅ Created {len(self.cv_splits)} fixed CV splits")
        for fold, (train_idx, val_idx) in enumerate(self.cv_splits):
            print(f"  Fold {fold+1}: Train={len(train_idx)}, Val={len(val_idx)}")
    
    def train_with_cv(self, features, labels, feature_indices, model_name="", epochs=100):
        """
        Train model with fixed CV splits.
        
        Args:
            features: Full feature matrix (31 features)
            labels: Labels
            feature_indices: Indices to use for this group
            model_name: Name for logging (pSLS, uSLS, cSLS, SLS)
            epochs: Number of epochs (paper uses 100)
        """
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(self.cv_splits):
            print(f"\n  Fold {fold+1}/{len(self.cv_splits)}")
            
            # Split data
            X_train_raw = features[train_idx][:, feature_indices]
            X_val_raw = features[val_idx][:, feature_indices]
            y_train = labels[train_idx]
            y_val = labels[val_idx]
            
            # Normalize per fold (as per paper)
            normalizer = FeatureNormalizer()
            X_train = normalizer.fit_transform(X_train_raw, [self.feature_names[i] for i in feature_indices])
            X_val = normalizer.transform(X_val_raw)
            
            # Create paper-exact SLS model with circle loss
            model = PaperExactSLS(
                input_dim=len(feature_indices),
                lstm_hidden=128,
                num_classes=2,
                dropout_rate=self.config.model.dropout_rate,
                se_reduction=self.config.model.se_reduction
            )
            
            # Trainer with circle loss (paper-exact)
            trainer = SLSTrainer(
                model=model,
                device=self.device,
                config=self.config.training.__dict__
            )
            
            # Data loaders
            train_loader = create_data_loader(
                X_train, y_train,
                batch_size=self.config.training.batch_size,
                add_channel_dim=True
            )
            val_loader = create_data_loader(
                X_val, y_val,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                add_channel_dim=True
            )
            
            # Train for fixed epochs (paper uses 100, no early stopping)
            best_metric, _ = trainer.train(
                train_loader,
                val_loader,
                epochs=epochs
            )
            
            # Evaluate on validation
            val_metrics, _ = trainer.validate(val_loader)
            
            fold_results.append({
                'fold': fold,
                'accuracy': val_metrics['accuracy'],
                'f1': val_metrics['f1'],
                'precision': val_metrics['precision'],
                'recall': val_metrics['recall']
            })
        
        # Aggregate results
        results = {
            'accuracy_mean': np.mean([r['accuracy'] for r in fold_results]),
            'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
            'f1_mean': np.mean([r['f1'] for r in fold_results]),
            'f1_std': np.std([r['f1'] for r in fold_results]),
            'num_features': len(feature_indices),
            'fold_results': fold_results
        }
        
        print(f"\n  {model_name} Results (mean ± std):")
        print(f"    Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
        print(f"    F1:       {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
        
        return results
    
    def study_feature_groups(self, epochs=100):
        """
        PAPER-EXACT: Compare pSLS, uSLS, cSLS, and full SLS.
        Reproduces Fig. 4 from the paper.
        """
        print("\n" + "=" * 80)
        print("FIGURE 4 REPRODUCTION: pSLS, uSLS, cSLS COMPARISON")
        print("=" * 80)
        
        results = {}
        
        # Test each feature group according to Table I
        test_cases = [
            ('propagation', 'pSLS'),
            ('user', 'uSLS'),
            ('content', 'cSLS'),
        ]
        
        for group_name, display_name in test_cases:
            indices = self.feature_groups[group_name]
            print(f"\n📊 Testing {display_name} ({len(indices)} features)")
            results[group_name] = self.train_with_cv(
                self.features_all, self.labels_all,
                feature_indices=indices,
                model_name=display_name,
                epochs=epochs
            )
        
        # Test with all 31 features (SLS)
        print(f"\n📊 Testing SLS (all 31 features)")
        results['sls'] = self.train_with_cv(
            self.features_all, self.labels_all,
            feature_indices=list(range(31)),
            model_name="SLS",
            epochs=epochs
        )
        
        # Print comparison table (paper style)
        print("\n" + "=" * 80)
        print("TABLE: Feature Group Comparison (mean ± std)")
        print("=" * 80)
        print(f"{'Model':6s} | {'Accuracy':20s} | {'F1':20s} | {'Features':10s}")
        print("-" * 80)
        
        for display, key in [('pSLS', 'propagation'), ('uSLS', 'user'), 
                            ('cSLS', 'content'), ('SLS', 'sls')]:
            if key in results:
                acc = results[key]['accuracy_mean']
                acc_std = results[key]['accuracy_std']
                f1 = results[key]['f1_mean']
                f1_std = results[key]['f1_std']
                n_feat = results[key]['num_features']
                print(f"{display:6s} | {acc:.4f} ± {acc_std:.4f} | {f1:.4f} ± {f1_std:.4f} | {n_feat:10d}")
        
        self.results['feature_groups'] = results
        return results
    
    def study_threshold_sweep(self, epochs=100):
        """
        PAPER-EXACT: Threshold sweep analysis.
        Reproduces Fig. 5 from the paper.
        
        Sweeps thresholds from 0.52 to 0.65 and reports:
        - Overall accuracy
        - SLS- accuracy (on uncertain subset)
        - GBDT accuracy (on uncertain subset)
        - Size of uncertain subset (dataset-)
        """
        print("\n" + "=" * 80)
        print("FIGURE 5 REPRODUCTION: THRESHOLD SWEEP ANALYSIS")
        print("=" * 80)
        print("Sweeping thresholds 0.52-0.65 as in paper Fig. 5")
        
        # Threshold range from paper (Fig. 5 shows 0.52-0.65)
        thresholds = np.arange(0.52, 0.66, 0.01)
        
        all_results = []
        
        for fold, (train_idx, val_idx) in enumerate(self.cv_splits):
            print(f"\n📊 Fold {fold+1}/{len(self.cv_splits)}")
            
            # Split data
            X_train_raw = self.features_all[train_idx]
            X_val_raw = self.features_all[val_idx]
            y_train = self.labels_all[train_idx]
            y_val = self.labels_all[val_idx]
            
            # Normalize
            normalizer = FeatureNormalizer()
            X_train = normalizer.fit_transform(X_train_raw, self.feature_names)
            X_val = normalizer.transform(X_val_raw)
            
            # Train SLS (paper-exact with circle loss)
            model = PaperExactSLS(
                input_dim=31,
                lstm_hidden=128,
                num_classes=2,
                dropout_rate=self.config.model.dropout_rate,
                se_reduction=self.config.model.se_reduction
            )
            
            trainer = SLSTrainer(
                model=model,
                device=self.device,
                config=self.config.training.__dict__
            )
            
            train_loader = create_data_loader(
                X_train, y_train,
                batch_size=self.config.training.batch_size,
                add_channel_dim=True
            )
            val_loader = create_data_loader(
                X_val, y_val,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                add_channel_dim=True
            )
            
            trainer.train(train_loader, val_loader, epochs=epochs)
            
            # Train GBDT on same training set
            gbdt = GBDTWrapper(**self.config.gbdt.__dict__)
            gbdt.fit(X_train, y_train)
            
            # Get SLS predictions on validation
            trainer.model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1).to(self.device)
                outputs = trainer.model(X_val_tensor)
                sls_probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            # Test each threshold
            fold_results = []
            for threshold in thresholds:
                sls_preds = np.argmax(sls_probs, axis=1)
                max_probs = np.max(sls_probs, axis=1)
                uncertain_mask = max_probs < threshold
                
                # Hybrid predictions
                final_preds = sls_preds.copy()
                if uncertain_mask.any():
                    gbdt_preds = gbdt.predict(X_val[uncertain_mask])
                    final_preds[uncertain_mask] = gbdt_preds
                
                # Calculate metrics
                metrics = Evaluator.compute_metrics(y_val, final_preds)
                
                # Calculate SLS- accuracy (on uncertain subset)
                sls_minus_acc = 0
                if uncertain_mask.any():
                    sls_minus_acc = (sls_preds[uncertain_mask] == y_val[uncertain_mask]).mean()
                
                # Calculate GBDT accuracy (on uncertain subset)
                gbdt_acc = 0
                if uncertain_mask.any():
                    gbdt_acc = (gbdt.predict(X_val[uncertain_mask]) == y_val[uncertain_mask]).mean()
                
                fold_results.append({
                    'fold': fold + 1,
                    'threshold': threshold,
                    'accuracy': metrics['accuracy'],
                    'f1': metrics['f1'],
                    'uncertain_count': uncertain_mask.sum(),
                    'uncertain_ratio': uncertain_mask.mean(),
                    'sls_minus_accuracy': sls_minus_acc,
                    'gbdt_accuracy': gbdt_acc,
                    'sls_accuracy': (sls_preds == y_val).mean()
                })
            
            all_results.extend(fold_results)
            
            # Print progress for this fold
            print(f"  Completed {len(thresholds)} thresholds")
        
        # Aggregate results
        df_results = pd.DataFrame(all_results)
        
        # Calculate mean and std across folds for each threshold
        agg_results = df_results.groupby('threshold').agg({
            'accuracy': ['mean', 'std'],
            'f1': ['mean', 'std'],
            'uncertain_ratio': 'mean',
            'sls_minus_accuracy': 'mean',
            'gbdt_accuracy': 'mean'
        }).reset_index()
        agg_results.columns = ['threshold', 'accuracy_mean', 'accuracy_std',
                              'f1_mean', 'f1_std', 'uncertain_ratio',
                              'sls_minus_accuracy', 'gbdt_accuracy']
        
        # Print summary table
        print("\n" + "=" * 80)
        print("THRESHOLD SWEEP RESULTS (mean across 5 folds)")
        print("=" * 80)
        print(f"{'Threshold':10s} | {'Accuracy':12s} | {'Uncertain':10s} | {'SLS- Acc':10s} | {'GBDT Acc':10s}")
        print("-" * 80)
        
        for _, row in agg_results.iterrows():
            print(f"{row['threshold']:.2f}       | {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | "
                  f"{row['uncertain_ratio']:.2%}    | {row['sls_minus_accuracy']:.4f}     | {row['gbdt_accuracy']:.4f}")
        
        # Paper's threshold 0.57 for reference
        paper_row = agg_results[agg_results['threshold'] == 0.57]
        if not paper_row.empty:
            print("\n" + "=" * 80)
            print("PAPER THRESHOLD (0.57) RESULTS")
            print("=" * 80)
            row = paper_row.iloc[0]
            print(f"Accuracy:          {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}")
            print(f"F1:                {row['f1_mean']:.4f} ± {row['f1_std']:.4f}")
            print(f"Uncertain ratio:   {row['uncertain_ratio']:.2%}")
            print(f"SLS- accuracy:     {row['sls_minus_accuracy']:.4f}")
            print(f"GBDT accuracy:     {row['gbdt_accuracy']:.4f}")
        
        self.results['threshold_sweep'] = agg_results
        return agg_results
    
    def plot_figure_4(self, save_dir="experiments/ablation_paper"):
        """Plot Figure 4: Feature group comparison."""
        if 'feature_groups' not in self.results:
            return
        
        results = self.results['feature_groups']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = ['pSLS', 'uSLS', 'cSLS', 'SLS']
        keys = ['propagation', 'user', 'content', 'sls']
        
        x = np.arange(len(models))
        width = 0.35
        
        accuracies = [results[k]['accuracy_mean'] for k in keys]
        acc_stds = [results[k]['accuracy_std'] for k in keys]
        f1_scores = [results[k]['f1_mean'] for k in keys]
        f1_stds = [results[k]['f1_std'] for k in keys]
        
        bars1 = ax.bar(x - width/2, accuracies, width, yerr=acc_stds,
                       label='Accuracy', capsize=5, alpha=0.8)
        bars2 = ax.bar(x + width/2, f1_scores, width, yerr=f1_stds,
                       label='F1 Score', capsize=5, alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Figure 4: Feature Group Ablation (pSLS, uSLS, cSLS)')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0.5, 1.0])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'figure_4_feature_ablation.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_figure_5(self, save_dir="experiments/ablation_paper"):
        """Plot Figure 5: Threshold sweep analysis."""
        if 'threshold_sweep' not in self.results:
            return
        
        df = self.results['threshold_sweep']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left plot: Accuracy vs Threshold (with uncertainty region)
        axes[0].fill_between(df['threshold'],
                            df['accuracy_mean'] - df['accuracy_std'],
                            df['accuracy_mean'] + df['accuracy_std'],
                            alpha=0.2, color='blue')
        axes[0].plot(df['threshold'], df['accuracy_mean'], 
                    'o-', linewidth=2, markersize=8, color='blue', label='Overall')
        axes[0].plot(df['threshold'], df['sls_minus_accuracy'], 
                    's--', linewidth=2, markersize=6, color='red', label='SLS- (uncertain)')
        axes[0].plot(df['threshold'], df['gbdt_accuracy'], 
                    '^--', linewidth=2, markersize=6, color='green', label='GBDT (uncertain)')
        axes[0].axvline(x=0.57, color='gray', linestyle=':', linewidth=2, label='Paper threshold (0.57)')
        axes[0].set_xlabel('Threshold')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Figure 5a: Accuracy vs Threshold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Right plot: Uncertain ratio vs Threshold
        axes[1].fill_between(df['threshold'], 0, df['uncertain_ratio'],
                            alpha=0.3, color='orange')
        axes[1].plot(df['threshold'], df['uncertain_ratio'], 
                    'o-', linewidth=2, markersize=8, color='orange')
        axes[1].axvline(x=0.57, color='gray', linestyle=':', linewidth=2)
        axes[1].set_xlabel('Threshold')
        axes[1].set_ylabel('Uncertain Ratio')
        axes[1].set_title('Figure 5b: Size of Dataset- vs Threshold')
        axes[1].grid(True, alpha=0.3)
        
        # Add threshold label
        paper_idx = df[df['threshold'] == 0.57].index
        if not paper_idx.empty:
            paper_ratio = df.loc[paper_idx, 'uncertain_ratio'].values[0]
            axes[1].plot(0.57, paper_ratio, 'ro', markersize=10)
            axes[1].annotate(f'{paper_ratio:.1%}', (0.57, paper_ratio), 
                           xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'figure_5_threshold_sweep.png'),
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_paper_ablation(self, epochs=100):
        """Run all paper-exact ablation studies."""
        print("\n" + "=" * 80)
        print("PAPER-EXACT ABLATION STUDIES")
        print("=" * 80)
        print("✓ 31 features only (exact Table I ordering)")
        print("✓ Feature groups: propagation(6), user(13), content(12)")
        print("✓ 5-fold stratified CV with fixed splits")
        print("✓ Per-fold normalization")
        print("✓ Circle loss")
        print("✓ Fixed epochs: 100 (no early stopping)")
        print("=" * 80)
        
        # Study 1: Feature group ablation (Fig. 4)
        self.study_feature_groups(epochs)
        
        # Study 2: Threshold sweep analysis (Fig. 5)
        self.study_threshold_sweep(epochs)
        
        # Generate paper figures
        save_dir = "experiments/ablation_paper"
        os.makedirs(save_dir, exist_ok=True)
        self.plot_figure_4(save_dir)
        self.plot_figure_5(save_dir)
    
    def save_results(self, save_dir="experiments/ablation_paper"):
        """Save ablation results."""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save feature group results
        if 'feature_groups' in self.results:
            rows = []
            for name, res in self.results['feature_groups'].items():
                display_name = 'pSLS' if name == 'propagation' else \
                              'uSLS' if name == 'user' else \
                              'cSLS' if name == 'content' else 'SLS'
                rows.append({
                    'model': display_name,
                    'accuracy_mean': res['accuracy_mean'],
                    'accuracy_std': res['accuracy_std'],
                    'f1_mean': res['f1_mean'],
                    'f1_std': res['f1_std'],
                    'num_features': res['num_features']
                })
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(save_dir, f'feature_groups_{timestamp}.csv'), index=False)
        
        # Save threshold sweep results
        if 'threshold_sweep' in self.results:
            self.results['threshold_sweep'].to_csv(
                os.path.join(save_dir, f'threshold_sweep_{timestamp}.csv'),
                index=False
            )
        
        print(f"\n✅ Results and figures saved to {save_dir}/")


def main():
    """Main function for paper-exact ablation study."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Paper-exact SLS ablation study')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Configuration file path')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to PHEME data directory')
    parser.add_argument('--save-dir', type=str, default='experiments/ablation_paper',
                       help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Epochs per fold (paper uses 100)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PAPER-EXACT SLS ABLATION STUDY")
    print("=" * 80)
    print("This script reproduces EXACTLY:")
    print("  - Table I feature grouping (31 features)")
    print("  - Fig. 4: pSLS, uSLS, cSLS comparison")
    print("  - Fig. 5: Threshold sweep analysis (0.52-0.65)")
    print("  - 5-fold stratified CV")
    print("  - Circle loss")
    print("=" * 80)
    
    # Initialize
    study = PaperExactAblation(args.config)
    
    # Load data
    events = study.load_pheme_data(args.data_dir)
    features, labels = study.extract_features_from_events(events)
    study.features_all = features
    study.labels_all = labels
    
    # Setup fixed CV splits
    study.setup_cross_validation(features, labels, n_folds=5)
    
    # Run paper ablation studies
    study.run_paper_ablation(epochs=args.epochs)
    
    # Save results
    study.save_results(args.save_dir)
    
    print("\n" + "=" * 80)
    print("PAPER-EXACT ABLATION STUDY COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()