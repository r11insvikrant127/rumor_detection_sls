"""
Main entry point for 56-feature rumor detection system.
"""

import argparse
import sys
from pathlib import Path
import traceback
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


# 🔴 FIX 2: More robust path setup
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Rumor Detection System with 56 Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --config configs/default.yaml --data data/train.csv
  python main.py predict --model experiments/best_model/ --data data/test.csv
  python main.py ablation --config configs/default.yaml --data data/train.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model with 56 features')
    train_parser.add_argument('--config', type=str, default='configs/default.yaml',
                            help='Configuration file path (default: configs/default.yaml)')
    train_parser.add_argument('--data', type=str, required=True,
                            help='Path to training data CSV file')
    train_parser.add_argument('--experiment', type=str, default=None,
                            help='Experiment name (default: auto-generated)')
    train_parser.add_argument('--epochs', type=int, default=None,
                            help='Number of epochs (overrides config)')
    train_parser.add_argument('--batch_size', type=int, default=None,
                            help='Batch size (overrides config)')
    train_parser.add_argument('--seed', type=int, default=None,
                            help='Random seed (overrides config)')
    train_parser.add_argument('--feature_set', type=str, choices=['all', 'paper', 'extended'],
                            default='all', help='Feature set to use (default: all 56 features)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', type=str, required=True,
                              help='Path to trained model directory or checkpoint')
    predict_parser.add_argument('--data', type=str, required=True,
                              help='Path to data CSV file for prediction')
    predict_parser.add_argument('--output', type=str, default='predictions.csv',
                              help='Output file for predictions (default: predictions.csv)')
    predict_parser.add_argument('--threshold', type=float, default=None,
                              help='Classification threshold (overrides config)')
    predict_parser.add_argument('--batch_size', type=int, default=32,
                              help='Batch size for prediction (default: 32)')
    predict_parser.add_argument('--feature_set', type=str, choices=['all', 'paper', 'extended'],
                              default='all', help='Feature set to use (default: all 56 features)')
    
    # Ablation study command
    ablation_parser = subparsers.add_parser('ablation', help='Run ablation study on feature groups')
    ablation_parser.add_argument('--config', type=str, default='configs/default.yaml',
                               help='Configuration file path')
    ablation_parser.add_argument('--data', type=str, required=True,
                               help='Path to training data CSV file')
    ablation_parser.add_argument('--output_dir', type=str, default='experiments/ablation',
                               help='Output directory for ablation results')
    ablation_parser.add_argument('--seed', type=int, default=42,
                               help='Random seed for reproducibility')
    ablation_parser.add_argument('--feature_groups', nargs='+',
                               default=['propagation', 'user', 'content', 'depth_breadth', 'optional'],
                               help='Feature groups to test (default: all)')
    
    # Feature extraction command
    extract_parser = subparsers.add_parser('extract', help='Extract 56 features from raw tweets')
    extract_parser.add_argument('--input', type=str, required=True,
                              help='Path to raw tweet data JSON file')
    extract_parser.add_argument('--output', type=str, required=True,
                              help='Output CSV file for extracted features')
    extract_parser.add_argument('--batch_size', type=int, default=100,
                              help='Number of events to process at once (default: 100)')
    extract_parser.add_argument('--feature_set', type=str, choices=['all', 'paper', 'extended'],
                              default='all', help='Feature set to extract (default: all 56 features)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the system components')
    test_parser.add_argument('--component', type=str, 
                           choices=['all', 'features', 'model', 'loss', 'config', 'trainer', 'depth_breadth'],
                           default='all', help='Component to test')
    test_parser.add_argument('--verbose', '-v', action='store_true',
                           help='Verbose output')
    test_parser.add_argument('--feature_set', type=str, choices=['all', 'paper', 'extended'],
                           default='all', help='Feature set to test (default: all 56 features)')
    
    # Config command (NEW - useful for debugging)
    config_parser = subparsers.add_parser('config', help='Show and validate configuration')
    config_parser.add_argument('--file', type=str, default='configs/default.yaml',
                             help='Config file to load')
    config_parser.add_argument('--print', action='store_true',
                             help='Print config summary')
    config_parser.add_argument('--validate', action='store_true',
                             help='Validate config')
    config_parser.add_argument('--save', type=str, default=None,
                             help='Save config to file')
    
    # Feature analysis command (NEW - implemented inline)
    analyze_parser = subparsers.add_parser('analyze', help='Analyze feature importance and statistics')
    analyze_parser.add_argument('--data', type=str, required=True,
                              help='Path to data CSV file')
    analyze_parser.add_argument('--output_dir', type=str, default='experiments/analysis',
                              help='Output directory for analysis results')
    analyze_parser.add_argument('--method', type=str, 
                              choices=['correlation', 'mutual_info', 'permutation', 'all'],
                              default='correlation',
                              help='Feature importance method')
    analyze_parser.add_argument('--top_k', type=int, default=20,
                              help='Number of top features to show (default: 20)')
    
    # Depth-breadth analysis command (NEW - implemented inline)
    depth_parser = subparsers.add_parser('depth_breadth', help='Analyze depth-breadth weighting methods')
    depth_parser.add_argument('--input', type=str, required=True,
                            help='Path to raw tweet data JSON file')
    depth_parser.add_argument('--output', type=str, default='depth_breadth_analysis.csv',
                            help='Output CSV file for analysis')
    depth_parser.add_argument('--sample_size', type=int, default=100,
                            help='Number of events to analyze (default: 100)')
    
    return parser.parse_args()


def run_feature_analysis(data_path, output_dir, method='correlation', top_k=20):
    """Analyze feature importance and statistics."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.inspection import permutation_importance
    
    print("=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Separate features and target (assuming last column is target)
    feature_cols = [col for col in df.columns if col != 'label' and col != 'event_id']
    if 'label' not in df.columns:
        print("Warning: No 'label' column found. Using last column as target.")
        target_col = df.columns[-1]
    else:
        target_col = 'label'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Basic statistics
    print("\n1. Basic Statistics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Class distribution:")
    class_counts = y.value_counts()
    for cls, count in class_counts.items():
        print(f"     Class {cls}: {count} ({count/len(y)*100:.1f}%)")
    
    # 2. Correlation analysis
    if method in ['correlation', 'all']:
        print("\n2. Correlation Analysis:")
        
        # Calculate correlation with target
        correlations = {}
        for col in feature_cols:
            corr = df[col].corr(df[target_col])
            correlations[col] = abs(corr)  # Use absolute value
        
        # Sort by correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        print(f"   Top {top_k} correlated features:")
        for i, (feature, corr) in enumerate(sorted_corr[:top_k]):
            print(f"     {i+1:3d}. {feature:30s}: {corr:.4f}")
        
        # Save correlation results
        corr_df = pd.DataFrame(sorted_corr, columns=['feature', 'correlation'])
        corr_df.to_csv(output_path / 'correlation_analysis.csv', index=False)
        
        # Plot correlation heatmap for top features
        if top_k <= 30:  # Only plot if manageable number of features
            plt.figure(figsize=(12, 10))
            top_features = [f for f, _ in sorted_corr[:top_k]] + [target_col]
            corr_matrix = df[top_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title(f'Correlation Matrix (Top {top_k} Features)')
            plt.tight_layout()
            plt.savefig(output_path / 'correlation_heatmap.png', dpi=150)
            plt.close()
        
        results['correlation'] = dict(sorted_corr)
    
    # 3. Mutual Information
    if method in ['mutual_info', 'all']:
        print("\n3. Mutual Information Analysis:")
        
        # Calculate mutual information
        try:
            mi_scores = mutual_info_classif(X, y)
            mi_features = list(zip(feature_cols, mi_scores))
            mi_features.sort(key=lambda x: x[1], reverse=True)
            
            print(f"   Top {top_k} features by mutual information:")
            for i, (feature, score) in enumerate(mi_features[:top_k]):
                print(f"     {i+1:3d}. {feature:30s}: {score:.4f}")
            
            # Save MI results
            mi_df = pd.DataFrame(mi_features, columns=['feature', 'mutual_information'])
            mi_df.to_csv(output_path / 'mutual_information_analysis.csv', index=False)
            
            # Plot MI scores
            plt.figure(figsize=(12, 6))
            top_mi_features = [f for f, _ in mi_features[:top_k]]
            top_mi_scores = [s for _, s in mi_features[:top_k]]
            
            plt.barh(range(len(top_mi_features)), top_mi_scores)
            plt.yticks(range(len(top_mi_features)), top_mi_features)
            plt.xlabel('Mutual Information')
            plt.title(f'Top {top_k} Features by Mutual Information')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(output_path / 'mutual_information_plot.png', dpi=150)
            plt.close()
            
            results['mutual_information'] = dict(mi_features)
        except Exception as e:
            print(f"   Warning: Mutual information calculation failed: {e}")
    
    # 4. Feature statistics summary
    print("\n4. Feature Statistics Summary:")
    stats_df = X.describe().T
    stats_df['missing'] = X.isnull().sum()
    stats_df['missing_pct'] = (X.isnull().sum() / len(X)) * 100
    
    print(f"   Statistics saved to {output_path / 'feature_statistics.csv'}")
    stats_df.to_csv(output_path / 'feature_statistics.csv')
    
    # Save summary report
    with open(output_path / 'analysis_summary.txt', 'w') as f:
        f.write("FEATURE ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Samples: {len(df)}\n")
        f.write(f"Features: {len(feature_cols)}\n")
        f.write(f"Target column: {target_col}\n")
        f.write(f"\nClass distribution:\n")
        for cls, count in class_counts.items():
            f.write(f"  Class {cls}: {count} ({count/len(y)*100:.1f}%)\n")
    
    print(f"\n✓ Analysis completed. Results saved to {output_dir}")


def analyze_depth_breadth_weights(input_path, output_path, sample_size=100):
    """Analyze depth-breadth weighting methods."""
    print("=" * 60)
    print("DEPTH-BREADTH WEIGHTING ANALYSIS")
    print("=" * 60)
    
    from src.preprocessing.feature_extractor import FeatureExtractor
    from src.preprocessing.tree_builder import TreeBuilder
    from src.preprocessing.kernel_subtree import KernelSubtreeExtractor
    
    # Load raw data
    with open(input_path, 'r') as f:
        raw_data = json.load(f)
    
    # Initialize components
    extractor = FeatureExtractor()
    tree_builder = TreeBuilder()
    kernel_extractor = KernelSubtreeExtractor()
    
    # Sample data if needed
    items = list(raw_data.items())
    if sample_size < len(items):
        import random
        items = random.sample(items, sample_size)
    
    print(f"Analyzing {len(items)} events...")
    
    results = []
    
    for event_id, event_data in tqdm(items, desc="Processing events"):
        try:
            # Build graph
            graph = tree_builder.build_from_tweets(event_data['tweets'])
            kernel_nodes = kernel_extractor.extract_kernel_subtree(graph)
            
            if not kernel_nodes:
                continue
            
            # Calculate weights using both methods
            weights1, depths1, breadths1 = extractor._compute_kernel_weights_method1(graph, kernel_nodes)
            weights2, depths2, breadths2 = extractor._compute_kernel_weights_method2(graph, kernel_nodes)
            
            # Calculate statistics
            stats = {
                'event_id': event_id,
                'kernel_size': len(kernel_nodes),
                'tree_depth': max(depths1) if depths1 else 0,
                'tree_nodes': len(graph.nodes()),
                'db1_mean_weight': float(np.mean(weights1)),
                'db1_std_weight': float(np.std(weights1)),
                'db1_mean_depth': float(np.mean(depths1)),
                'db1_mean_breadth': float(np.mean(breadths1)),
                'db2_mean_weight': float(np.mean(weights2)),
                'db2_std_weight': float(np.std(weights2)),
                'db2_mean_depth': float(np.mean(depths2)),
                'db2_mean_breadth': float(np.mean(breadths2)),
                'weight_correlation': float(np.corrcoef(weights1, weights2)[0, 1]) if len(weights1) > 1 else 0.0
            }
            
            results.append(stats)
            
        except Exception as e:
            print(f"Warning: Failed to analyze event {event_id}: {e}")
    
    # Create results DataFrame
    if results:
        results_df = pd.DataFrame(results)
        
        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Analysis completed. Results saved to {output_path}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"  Events analyzed: {len(results)}")
        print(f"  Average kernel size: {results_df['kernel_size'].mean():.1f}")
        print(f"  Average tree depth: {results_df['tree_depth'].mean():.1f}")
        print(f"  Method 1 mean weight: {results_df['db1_mean_weight'].mean():.3f}")
        print(f"  Method 2 mean weight: {results_df['db2_mean_weight'].mean():.3f}")
        print(f"  Weight correlation: {results_df['weight_correlation'].mean():.3f}")
        
        # Print correlation between methods
        print("\nMethod Comparison:")
        for col in ['kernel_size', 'tree_depth']:
            corr1 = results_df['db1_mean_weight'].corr(results_df[col])
            corr2 = results_df['db2_mean_weight'].corr(results_df[col])
            print(f"  {col:15s}: Method1 correlation={corr1:.3f}, Method2 correlation={corr2:.3f}")
    else:
        print("✗ No results to save. Check your input data.")


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == 'train':
        # 🔴 FIX 3: Pass arguments directly instead of mutating sys.argv
        # This is cleaner and doesn't mutate global state
        from src.scripts.train import train
        
        # Prepare kwargs for train function
        train_kwargs = {
            'config_path': args.config,
            'data_path': args.data,
            'experiment_name': args.experiment,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'seed': args.seed,
            'feature_set': args.feature_set
        }
        
        # Remove None values
        train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}
        
        # Call train function
        train(**train_kwargs)
        
    elif args.command == 'predict':
        from src.scripts.predict import predict
        
        predict_kwargs = {
            'model_path': args.model,
            'data_path': args.data,
            'output_path': args.output,
            'threshold': args.threshold,
            'batch_size': args.batch_size,
            'feature_set': args.feature_set
        }
        
        predict_kwargs = {k: v for k, v in predict_kwargs.items() if v is not None}
        predict(**predict_kwargs)
        
    elif args.command == 'ablation':
        from src.scripts.ablation_study import run_ablation_study
        
        ablation_kwargs = {
            'config_path': args.config,
            'data_path': args.data,
            'output_dir': args.output_dir,
            'seed': args.seed,
            'feature_groups': args.feature_groups
        }
        
        run_ablation_study(**ablation_kwargs)
        
    elif args.command == 'extract':
        from src.preprocessing.feature_extractor import FeatureExtractor
        import pandas as pd
        import json
        from tqdm import tqdm
        
        print(f"Extracting features from {args.input}...")
        print(f"Feature set: {args.feature_set}")
        
        # Load raw data
        with open(args.input, 'r') as f:
            raw_data = json.load(f)
        
        # Initialize extractor
        extractor = FeatureExtractor()
        all_feature_names = extractor.get_feature_names()
        
        # Select feature set
        if args.feature_set == 'paper':
            # Get only paper features (1-31)
            paper_feature_indices = list(range(31))  # First 31 features are paper features
            feature_names = [all_feature_names[i] for i in paper_feature_indices]
        elif args.feature_set == 'extended':
            # Get paper + optional features (1-47, excluding depth-breadth)
            extended_indices = list(range(47))  # First 47 features (paper + optional)
            feature_names = [all_feature_names[i] for i in extended_indices]
        else:  # 'all'
            feature_names = all_feature_names
        
        print(f"✓ Feature extractor initialized with {len(feature_names)} features")
        
        # Extract features in batches
        features = []
        event_ids = []
        errors = []
        
        items = list(raw_data.items())
        for i in tqdm(range(0, len(items), args.batch_size), desc="Processing events"):
            batch = items[i:i + args.batch_size]
            
            for event_id, event_data in batch:
                try:
                    feature_vector = extractor.extract_features(event_data)
                    
                    # Select features based on feature_set
                    if args.feature_set == 'paper':
                        feature_vector = feature_vector[:31]
                    elif args.feature_set == 'extended':
                        feature_vector = feature_vector[:47]
                    # 'all' uses all features
                    
                    features.append(feature_vector)
                    event_ids.append(event_id)
                except Exception as e:
                    errors.append((event_id, str(e)))
                    if args.verbose:
                        print(f"✗ Error extracting features for event {event_id}: {e}")
        
        # Create DataFrame
        df = pd.DataFrame(features, columns=feature_names)
        df.insert(0, 'event_id', event_ids)
        
        # Save to CSV
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Extracted {len(features)} events with {len(feature_names)} features")
        print(f"✓ Saved to {args.output}")
        
        if errors:
            print(f"\n⚠️  Encountered {len(errors)} errors during extraction")
            if args.verbose:
                for event_id, error in errors[:10]:  # Show first 10 errors
                    print(f"  - {event_id}: {error}")
                if len(errors) > 10:
                    print(f"  ... and {len(errors) - 10} more errors")
        
        # Print feature summary
        print(f"\nFeature summary:")
        print(f"  Total features: {len(feature_names)}")
        
        # Count feature categories
        if args.feature_set == 'all':
            print(f"  Paper features (1-31): 31")
            print(f"  Additional metrics: 7")
            print(f"  Optional features: 4")
            print(f"  Depth-breadth features: 13")
            print(f"  Total: 56 features")
        elif args.feature_set == 'paper':
            print(f"  Propagation features: 6")
            print(f"  User features: 13 (6 source + 7 kernel)")
            print(f"  Content features: 12 (2 source + 10 kernel)")
            print(f"  Total paper features: 31")
        elif args.feature_set == 'extended':
            print(f"  Paper features: 31")
            print(f"  Additional metrics: 7")
            print(f"  Optional features: 4")
            print(f"  Backward compatibility: 3")
            print(f"  Total extended features: 47")
        
    elif args.command == 'test':
        from src.utils.helpers import set_seed
        set_seed(42)
        
        if args.component in ['all', 'features']:
            print("=" * 60)
            print("TESTING FEATURE EXTRACTOR")
            print("=" * 60)
            
            from src.preprocessing.feature_extractor import FeatureExtractor
            extractor = FeatureExtractor()
            feature_names = extractor.get_feature_names()
            
            print(f"✓ Number of features: {len(feature_names)}")
            print(f"✓ Feature set: {args.feature_set}")
            
            if args.verbose:
                print("\nAll 56 features:")
                for i, name in enumerate(feature_names):
                    category = "Paper" if i < 31 else "Extended" if i < 47 else "Depth-Breadth"
                    print(f"  {i:3d} [{category:15s}]: {name}")
                
                # Print summary by category
                print(f"\nFeature breakdown:")
                print(f"  Paper features (1-31): {sum(1 for i in range(len(feature_names)) if i < 31)}")
                print(f"  Additional metrics: {sum(1 for i, name in enumerate(feature_names) if 31 <= i < 38)}")
                print(f"  Optional features: {sum(1 for i, name in enumerate(feature_names) if 38 <= i < 42)}")
                print(f"  Depth-breadth features: {sum(1 for i, name in enumerate(feature_names) if i >= 42)}")
        
        if args.component in ['all', 'depth_breadth']:
            print("\n" + "=" * 60)
            print("TESTING DEPTH-BREADTH WEIGHTING METHODS")
            print("=" * 60)
            
            # Create a test graph to analyze depth-breadth weighting
            import networkx as nx
            
            # Create a sample tree
            G = nx.DiGraph()
            # Add root node
            G.add_node("root", user={"followers_count": 1000}, text="Root tweet")
            
            # Add level 1 nodes
            for i in range(3):
                node_id = f"level1_{i}"
                G.add_node(node_id, user={"followers_count": 500 + i*100}, text=f"Level 1 tweet {i}")
                G.add_edge("root", node_id)
                
                # Add level 2 nodes
                for j in range(2):
                    node_id2 = f"level2_{i}_{j}"
                    G.add_node(node_id2, user={"followers_count": 200 + j*50}, text=f"Level 2 tweet {i}_{j}")
                    G.add_edge(node_id, node_id2)
            
            # Test depth-breadth weighting
            from src.preprocessing.feature_extractor import FeatureExtractor
            extractor = FeatureExtractor()
            
            # Get kernel nodes (all nodes for testing)
            kernel_nodes = list(G.nodes())
            
            # Test both methods
            weights1, depths1, breadths1 = extractor._compute_kernel_weights_method1(G, kernel_nodes)
            weights2, depths2, breadths2 = extractor._compute_kernel_weights_method2(G, kernel_nodes)
            
            print(f"✓ Method 1 weights: mean={np.mean(weights1):.3f}, std={np.std(weights1):.3f}")
            print(f"✓ Method 2 weights: mean={np.mean(weights2):.3f}, std={np.std(weights2):.3f}")
            
            if args.verbose:
                print("\nDetailed weight comparison (first 5 nodes):")
                print(f"{'Node':15s} {'Depth1':>8s} {'Breadth1':>8s} {'Weight1':>8s} {'Depth2':>8s} {'Breadth2':>8s} {'Weight2':>8s}")
                for i in range(min(5, len(kernel_nodes))):
                    print(f"{kernel_nodes[i]:15s} {depths1[i]:8.1f} {breadths1[i]:8.1f} {weights1[i]:8.3f} {depths2[i]:8.1f} {breadths2[i]:8.1f} {weights2[i]:8.3f}")
        
        if args.component in ['all', 'loss']:
            print("\n" + "=" * 60)
            print("TESTING LOSS FUNCTIONS")
            print("=" * 60)
            
            import torch
            from src.training.loss import create_loss_for_rumor_detection
            
            # Test different loss functions
            loss_types = ['circle', 'combined', 'focal', 'weighted_circle']
            batch_size = 4
            num_classes = 2
            
            logits = torch.randn(batch_size, num_classes)
            labels = torch.randint(0, num_classes, (batch_size,))
            
            for loss_type in loss_types:
                try:
                    loss_fn = create_loss_for_rumor_detection(loss_type=loss_type)
                    loss = loss_fn(logits, labels)
                    print(f"✓ {loss_type.upper():15s} loss: {loss.item():.6f}")
                except Exception as e:
                    print(f"✗ {loss_type.upper():15s} loss failed: {e}")
        
        if args.component in ['all', 'model']:
            print("\n" + "=" * 60)
            print("TESTING MODEL ARCHITECTURE")
            print("=" * 60)
            
            from src.models.sls import SLSModel
            
            try:
                # Test with different input dimensions based on feature set
                if args.feature_set == 'paper':
                    input_dim = 31
                elif args.feature_set == 'extended':
                    input_dim = 47
                else:  # 'all'
                    input_dim = 56
                
                model = SLSModel(input_dim=input_dim, num_classes=2)
                print(f"✓ Model created with input_dim={input_dim} ({args.feature_set} features)")
                print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
                
                # Test forward pass
                import torch
                test_input = torch.randn(2, 1, input_dim)
                output = model(test_input)
                print(f"✓ Forward pass works: input={test_input.shape}, output={output.shape}")
            except Exception as e:
                print(f"✗ Model test failed: {e}")
                traceback.print_exc()


        if args.component in ['all', 'trainer']:
            print("\n" + "=" * 60)
            print("TESTING TRAINER")
            print("=" * 60)
            
            try:
                from src.training.trainer import SLSTrainer
                from src.models.sls import SLSModel
                
                # Determine input dimension
                if args.feature_set == 'paper':
                    input_dim = 31
                elif args.feature_set == 'extended':
                    input_dim = 47
                else:
                    input_dim = 56
                
                model = SLSModel(input_dim=input_dim, num_classes=2)
                
                # Use correct parameter names for CombinedLoss
                trainer_config = {
                    'learning_rate': 0.001,
                    'loss_type': 'combined',
                    'num_classes': 2,
                    'class_weights': [0.3, 0.7],
                    'margin': 0.25,
                    'gamma_circle': 256, 
                    'gamma_focal': 2.0,
                    'alpha_focal': 0.25,
                    'weight_circle': 0.7,
                    'label_smoothing': 0.1,
                    'patience': 10
                }
                
                trainer = SLSTrainer(model, device='cpu', config=trainer_config)
                print(f"✓ Trainer initialized successfully")
                print(f"✓ Using {input_dim} features ({args.feature_set} feature set)")
                print(f"✓ Loss function: {type(trainer.criterion).__name__}")
                
            except Exception as e:
                print(f"✗ Trainer test failed: {e}")
                print("\nDebug info: The CombinedLoss expects 'gamma_circle' not 'gamma'")
                print("Check the loss_kwargs in trainer.py")
                if args.verbose:
                    traceback.print_exc()
        
        if args.component in ['all', 'config']:
            print("\n" + "=" * 60)
            print("TESTING CONFIGURATION")
            print("=" * 60)
            
            try:
                from src.utils.config import ConfigManager
                
                config = ConfigManager()
                print(f"✓ Config loaded")
                print(f"✓ Model input dimension: {config.model.input_dim} features")
                print(f"✓ Loss type: {config.loss.loss_type}")
                print(f"✓ Training epochs: {config.training.epochs}")
                
                # Test feature groups
                print(f"\nFeature groups:")
                for group, indices in config.feature_groups.items():
                    print(f"  {group}: {len(indices)} features")
            except Exception as e:
                print(f"✗ Config test failed: {e}")
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        
    elif args.command == 'config':
        from src.utils.config import ConfigManager
        
        config = ConfigManager(args.file if hasattr(args, 'file') else 'configs/default.yaml')
        
        if args.validate:
            print("✓ Configuration validated successfully")
        
        if args.print:
            config.print_summary()
        
        if args.save:
            config.save(args.save)
            print(f"✓ Configuration saved to {args.save}")
    
    elif args.command == 'analyze':
        run_feature_analysis(
            data_path=args.data,
            output_dir=args.output_dir,
            method=args.method,
            top_k=args.top_k
        )
    
    elif args.command == 'depth_breadth':
        analyze_depth_breadth_weights(
            input_path=args.input,
            output_path=args.output,
            sample_size=args.sample_size
        )
        
    else:
        print("Error: No command specified")
        print("\nAvailable commands:")
        print("  train         - Train the model with 56 features")
        print("  predict       - Make predictions")
        print("  ablation      - Run ablation study on feature groups")
        print("  extract       - Extract features from raw data")
        print("  test          - Test system components")
        print("  config        - Show and validate configuration")
        print("  analyze       - Analyze feature importance")
        print("  depth_breadth - Analyze depth-breadth weighting")
        print("\nUse 'python main.py <command> --help' for more information")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if '--verbose' in sys.argv or '-v' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)