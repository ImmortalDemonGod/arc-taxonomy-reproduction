import argparse
import json
from pathlib import Path
from collections import defaultdict
import sys

import numpy as np
from safetensors.torch import save_file

# Make reproduction/src importable as 'src'
REPRO_DIR = Path(__file__).resolve().parent.parent
if str(REPRO_DIR) not in sys.path:
    sys.path.insert(0, str(REPRO_DIR))

# Use centralized utilities within reproduction package
from src.lora_utils import flatten_adapter, average_adapters


CATEGORY_ORDER = [
    "S1", "S2", "S3",
    "C1", "C2",
    "K1", "L1",
    "A1", "A2",
]


def find_any_adapter_vector_dim(atomic_dir: Path) -> int:
    for p in atomic_dir.rglob("adapter_model.safetensors"):
        v = flatten_adapter(str(p.parent))
        return int(v.numel())
    raise FileNotFoundError(f"No adapters found under {atomic_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=str,
        default="../../../../data/taxonomy_classification/all_tasks_classified.json",
        help="Path to v3 labels JSON (task_id -> category)",
    )
    parser.add_argument(
        "--atomic-dir",
        type=str,
        default="outputs/atomic_loras",
        help="Directory containing atomic LoRA adapters (task_id/adapter_model.safetensors)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visual_classifier",
        help="Output directory for centroids and merged adapters",
    )
    parser.add_argument(
        "--no-merged-adapters",
        action="store_true",
        help="If set, do not save merged category adapters",
    )
    args = parser.parse_args()

    # Resolve paths robustly: prefer user-provided, else derive from script location
    provided_labels = Path(args.labels)
    if provided_labels.is_absolute() and provided_labels.exists():
        labels_path = provided_labels
    else:
        # Try relative to CWD
        labels_path = provided_labels.resolve()
        if not labels_path.exists():
            # Fallback: compute from script location up to arc_reactor/data
            script_path = Path(__file__).resolve()
            # parents: 0=scripts, 1=reproduction, 2=arc_taxonomy_2025, 3=publications, 4=arc_reactor
            try:
                arc_reactor_dir = script_path.parents[4]
                labels_path = arc_reactor_dir / 'data' / 'taxonomy_classification' / 'all_tasks_classified.json'
            except IndexError:
                pass

    # Atomic and output dirs: resolve relative to reproduction/ by default
    atomic_dir = Path(args.atomic_dir)
    if not atomic_dir.is_absolute():
        atomic_dir = (REPRO_DIR / atomic_dir)
    atomic_dir = atomic_dir.resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPRO_DIR / output_dir)
    output_dir = output_dir.resolve()
    merged_dir = output_dir / "category_skills_v3"
    centroids_path = output_dir / "category_centroids_v3.npy"

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_path) as f:
        all_labels = json.load(f)

    category_to_tasks = defaultdict(list)
    for task_id, cat in all_labels.items():
        if cat == "ambiguous":
            continue
        category_to_tasks[cat].append(task_id)

    # Determine vector dimension D from any available adapter
    D = find_any_adapter_vector_dim(atomic_dir)

    centroids = []
    summary = []

    for cat in CATEGORY_ORDER:
        task_ids = category_to_tasks.get(cat, [])
        adapter_paths = []
        for t in task_ids:
            ap = atomic_dir / t / "adapter_model.safetensors"
            if ap.exists():
                adapter_paths.append(str(ap.parent))
        used = len(adapter_paths)
        total = len(task_ids)
        coverage = (used / total) if total else 0.0
        print(f"{cat}: using {used}/{total} adapters (coverage={coverage:.1%})")

        if used == 0:
            centroids.append(np.zeros(D, dtype=np.float32))
        else:
            vecs = [flatten_adapter(p).numpy() for p in adapter_paths]
            centroid = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
            centroids.append(centroid)

            if not args.no_merged_adapters:
                avg_sd = average_adapters(adapter_paths)
                out_path = merged_dir / f"{cat.lower()}_adapter.safetensors"
                save_file(avg_sd, out_path)

        summary.append({
            "category": cat,
            "total_tasks": total,
            "found_adapters": used,
            "coverage": coverage,
        })

    centroids_arr = np.stack(centroids, axis=0)
    np.save(centroids_path, centroids_arr)

    print("\nSaved centroids:", centroids_path)
    print("Shape:", centroids_arr.shape)

    print("\nCoverage Summary:")
    print(f"{'Category':<8} {'Found/Total':<15} {'Coverage':<9}")
    for s in summary:
        print(f"{s['category']:<8} {s['found_adapters']}/{s['total_tasks']:<13} {s['coverage']:.1%}")


if __name__ == "__main__":
    main()
