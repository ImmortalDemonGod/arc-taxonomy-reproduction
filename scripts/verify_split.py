#!/usr/bin/env python3
"""
Verify the quality of train/val split for distributional_alignment dataset.

Checks:
1. Category balance (each category ~80/20 train/val)
2. Output size distribution similarity (train vs val)
3. No degenerate task bias in validation set
"""
import json
from pathlib import Path
from collections import defaultdict
import statistics


def get_output_sizes(task_files: list, data_dir: Path) -> list:
    """Get mean output sizes for a list of task files."""
    sizes = []
    
    for filename in task_files:
        task_id = filename.replace('.json', '')
        task_file = data_dir / filename
        
        try:
            with open(task_file) as f:
                task_data = json.load(f)
            
            task_sizes = []
            for ex in task_data.get('train', []):
                output = ex.get('output', [])
                h = len(output)
                w = len(output[0]) if output else 0
                task_sizes.append(h * w)
            
            if task_sizes:
                sizes.append(statistics.mean(task_sizes))
                
        except Exception as e:
            print(f"Warning: Error reading {filename}: {e}")
            continue
    
    return sizes


def verify_split(
    split_manifest_file: Path,
    task_categories_file: Path,
    data_dir: Path
):
    """
    Verify split quality with comprehensive checks.
    
    Args:
        split_manifest_file: Path to split_manifest.json
        task_categories_file: Path to task_categories.json
        data_dir: Directory containing task JSON files
    """
    print("="*70)
    print("SPLIT VERIFICATION REPORT")
    print("="*70)
    
    # Load files
    with open(split_manifest_file) as f:
        manifest = json.load(f)
    
    with open(task_categories_file) as f:
        categories = json.load(f)
    
    train_files = manifest['train_files']
    val_files = manifest['val_files']
    
    # Extract task IDs
    train_ids = [f.replace('.json', '') for f in train_files]
    val_ids = [f.replace('.json', '') for f in val_files]
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total tasks: {manifest['total_tasks']}")
    print(f"  Train tasks: {manifest['train_tasks']} ({manifest['train_examples']:,} examples)")
    print(f"  Val tasks: {manifest['val_tasks']} ({manifest['val_examples']:,} examples)")
    print(f"  Train ratio: {manifest['train_tasks']/manifest['total_tasks']*100:.1f}%")
    
    # Check 1: Category balance
    print(f"\n✅ CHECK 1: Category Balance")
    print(f"{'Category':<12} {'Train':<6} {'Val':<6} {'Total':<6} {'Train %':<8} {'Status':<10}")
    print("-" * 70)
    
    category_train = defaultdict(int)
    category_val = defaultdict(int)
    
    for task_id in train_ids:
        category = categories.get(task_id, 'unknown')
        category_train[category] += 1
    
    for task_id in val_ids:
        category = categories.get(task_id, 'unknown')
        category_val[category] += 1
    
    all_categories = set(category_train.keys()) | set(category_val.keys())
    balance_issues = []
    
    for category in sorted(all_categories):
        train_count = category_train[category]
        val_count = category_val[category]
        total = train_count + val_count
        train_pct = (train_count / total * 100) if total > 0 else 0
        
        # Check if within acceptable range (75-85%)
        status = "✅ PASS" if 75 <= train_pct <= 85 else "❌ FAIL"
        if status == "❌ FAIL":
            balance_issues.append(category)
        
        print(f"{category:<12} {train_count:<6} {val_count:<6} {total:<6} {train_pct:<8.1f} {status:<10}")
    
    # Check 2: Output size distribution
    print(f"\n✅ CHECK 2: Output Size Distribution")
    
    train_sizes = get_output_sizes(train_files, data_dir)
    val_sizes = get_output_sizes(val_files, data_dir)
    
    if train_sizes and val_sizes:
        print(f"{'Split':<10} {'Count':<8} {'Mean':<10} {'Median':<10} {'Min':<8} {'Max':<8}")
        print("-" * 70)
        print(f"{'Train':<10} {len(train_sizes):<8} {statistics.mean(train_sizes):<10.1f} {statistics.median(train_sizes):<10.1f} {min(train_sizes):<8.1f} {max(train_sizes):<8.1f}")
        print(f"{'Val':<10} {len(val_sizes):<8} {statistics.mean(val_sizes):<10.1f} {statistics.median(val_sizes):<10.1f} {min(val_sizes):<8.1f} {max(val_sizes):<8.1f}")
        
        # Simplified distribution similarity check
        mean_diff_pct = abs(statistics.mean(train_sizes) - statistics.mean(val_sizes)) / statistics.mean(train_sizes) * 100
        median_diff_pct = abs(statistics.median(train_sizes) - statistics.median(val_sizes)) / statistics.median(train_sizes) * 100
        
        print(f"\n  Mean difference: {mean_diff_pct:.1f}%")
        print(f"  Median difference: {median_diff_pct:.1f}%")
        
        size_distribution_ok = mean_diff_pct < 20 and median_diff_pct < 20
        print(f"  Status: {'✅ PASS' if size_distribution_ok else '❌ FAIL'} (should be < 20% difference)")
    else:
        print("  ❌ Could not compute size distributions")
        size_distribution_ok = False
    
    # Check 3: No degenerate task bias
    print(f"\n✅ CHECK 3: Degenerate Task Distribution")
    print(f"{'Category':<12} {'Split':<8} {'Tasks':<8} {'Tiny (<10)':<12} {'Tiny %':<10} {'Status':<10}")
    print("-" * 70)
    
    degenerate_issues = []
    
    for category in ['A1', 'ambiguous']:
        # Train
        cat_train_files = [f for tid, f in zip(train_ids, train_files) if categories.get(tid) == category]
        cat_train_sizes = get_output_sizes(cat_train_files, data_dir)
        
        if cat_train_sizes:
            train_tiny = sum(1 for s in cat_train_sizes if s < 10)
            train_tiny_pct = train_tiny / len(cat_train_sizes) * 100
            train_status = "✅ PASS" if train_tiny_pct < 50 else "⚠️  WARN"
            print(f"{category:<12} {'Train':<8} {len(cat_train_sizes):<8} {train_tiny:<12} {train_tiny_pct:<10.1f} {train_status:<10}")
        
        # Val
        cat_val_files = [f for tid, f in zip(val_ids, val_files) if categories.get(tid) == category]
        cat_val_sizes = get_output_sizes(cat_val_files, data_dir)
        
        if cat_val_sizes:
            val_tiny = sum(1 for s in cat_val_sizes if s < 10)
            val_tiny_pct = val_tiny / len(cat_val_sizes) * 100
            val_status = "✅ PASS" if val_tiny_pct < 50 else "❌ FAIL"
            
            if val_status == "❌ FAIL":
                degenerate_issues.append(category)
            
            print(f"{category:<12} {'Val':<8} {len(cat_val_sizes):<8} {val_tiny:<12} {val_tiny_pct:<10.1f} {val_status:<10}")
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = len(balance_issues) == 0 and size_distribution_ok and len(degenerate_issues) == 0
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print("\nThe split is high-quality and ready for training.")
    else:
        print("\n❌ SOME CHECKS FAILED:")
        if balance_issues:
            print(f"  - Category balance issues: {', '.join(balance_issues)}")
        if not size_distribution_ok:
            print(f"  - Size distribution differs significantly between train/val")
        if degenerate_issues:
            print(f"  - Degenerate task bias in validation: {', '.join(degenerate_issues)}")
        print("\n⚠️  Consider regenerating split with different random seed or adjusting stratification.")
    
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "distributional_alignment"
    manifest_file = data_dir / "split_manifest.json"
    categories_file = data_dir / "task_categories.json"
    
    # Run verification
    passed = verify_split(manifest_file, categories_file, data_dir)
    
    # Exit with appropriate code
    exit(0 if passed else 1)
