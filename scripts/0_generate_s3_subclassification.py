#!/usr/bin/env python3
"""
Create final S3 sub-classification by combining:
1. Automated heuristic classifications (works for 101/108 tasks)
2. Manual classifications for remaining ambiguous tasks (7 tasks)

This creates the production-ready S3-A vs S3-B taxonomy.
"""

import json
from pathlib import Path

# Load automated classifier results
automated_file = Path(__file__).parent.parent / "data/taxonomy/s3_subclassification_results.json"
with open(automated_file, 'r') as f:
    automated = json.load(f)

# Manual classifications for the ambiguous tasks
manual_classifications = {
    '673ef223': 'S3-A',  # Loop with independent connects to edges
    '6d58a25d': 'S3-B',  # Filtered multi-source connects with visibility query
    '8d510a79': 'S3-A',  # Loop with independent conditional connects
    'a2fd1cf0': 'S3-A',  # Fixed 2-point L-shaped connection
    'd43fd935': 'S3-A',  # Simple topological operations
    'e9614598': 'S3-A',  # Pattern-based primitives
    'f15e1fac': 'S3-A'   # Pattern-based connections
}

print("=" * 80)
print("CREATING FINAL S3 SUB-CLASSIFICATION")
print("=" * 80)

final_results = []
manual_count = 0
automated_count = 0

for task in automated:
    task_id = task['task_id']
    
    # Check if we have a manual classification
    if task_id in manual_classifications:
        # Override with manual classification
        task['classification'] = manual_classifications[task_id]
        task['confidence'] = 0.95  # High confidence from manual review
        task['manual_override'] = True
        manual_count += 1
        print(f"✓ {task_id}: Manual override → {task['classification']}")
    else:
        task['manual_override'] = False
        automated_count += 1
    
    final_results.append(task)

print(f"\nTotal tasks: {len(final_results)}")
print(f"  Automated: {automated_count}")
print(f"  Manual: {manual_count}")

# Count final distribution
s3a_count = sum(1 for r in final_results if r['classification'] == 'S3-A')
s3b_count = sum(1 for r in final_results if r['classification'] == 'S3-B')
ambig_count = sum(1 for r in final_results if r['classification'] == 'S3-AMBIGUOUS')

print("\n" + "=" * 80)
print("FINAL S3 DISTRIBUTION")
print("=" * 80)
print(f"\nTotal S3 tasks: {len(final_results)}")
print(f"  S3-A (Pattern-based):  {s3a_count:3d} ({s3a_count/len(final_results)*100:5.1f}%)")
print(f"  S3-B (Graph reasoning): {s3b_count:3d} ({s3b_count/len(final_results)*100:5.1f}%)")
print(f"  Ambiguous:             {ambig_count:3d} ({ambig_count/len(final_results)*100:5.1f}%)")

print(f"\nOf 400 total curriculum tasks:")
print(f"  S3-A: {s3a_count} ({s3a_count/4:.1f}%) - Transformer-friendly")
print(f"  S3-B: {s3b_count} ({s3b_count/4:.1f}%) - Need GNN/graph-aware modifications")

# Save final results
output_file = Path(__file__).parent.parent / "data/taxonomy/s3_final_classification.json"
with open(output_file, 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"\nFinal classification saved to:")
print(f"  {output_file}")

# Create simple lookup dict
s3_lookup = {}
for task in final_results:
    s3_lookup[task['task_id']] = {
        'classification': task['classification'],
        'confidence': task['confidence'],
        'manual': task.get('manual_override', False)
    }

lookup_file = Path(__file__).parent.parent / "data/taxonomy/s3_lookup.json"
with open(lookup_file, 'w') as f:
    json.dump(s3_lookup, f, indent=2)

print(f"  {lookup_file}")

print("\n" + "=" * 80)
print("TAXONOMY COMPLETE!")
print("=" * 80)
print("\n✅ All 108 S3 tasks classified:")
print(f"   - {s3a_count} S3-A (pattern-based, Transformer-friendly)")
print(f"   - {s3b_count} S3-B (graph reasoning, needs GNN)")
print(f"   - {ambig_count} ambiguous (if any remaining)")

if ambig_count == 0:
    print("\n🎯 PERFECT! No ambiguous tasks remaining!")
    print("   Taxonomy is production-ready.")
