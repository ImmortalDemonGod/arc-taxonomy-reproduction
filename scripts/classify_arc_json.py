#!/usr/bin/env python3
"""
Classify ARC(-AGI-2) JSON tasks into taxonomy categories using TaskEncoderCNN.

Example:
  python scripts/classify_arc_json.py \
    --input-dir data/arc_agi_temp \
    --checkpoint outputs/visual_classifier/models/task_encoder_direct_v3_best.pt \
    --centroids outputs/visual_classifier/category_centroids_v3.npy \
    --output outputs/visual_classifier/results/arc2_classify_seed.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make reproduction/src importable as 'src'
REPRO_DIR = Path(__file__).resolve().parent.parent
if str(REPRO_DIR) not in sys.path:
    sys.path.insert(0, str(REPRO_DIR))

from src.models.task_encoder_cnn import TaskEncoderCNN  # noqa: E402


DEFAULT_CATS = [
    "S1", "S2", "S3",
    "C1", "C2",
    "K1", "L1",
    "A1", "A2",
]


def pad_grid(grid: torch.Tensor, max_size: int = 30, pad_token: int = 10) -> torch.Tensor:
    h, w = grid.shape
    if h > max_size or w > max_size:
        grid = grid[:max_size, :max_size]
        h, w = grid.shape
    pad_h = max_size - h
    pad_w = max_size - w
    return F.pad(grid, (0, pad_w, 0, pad_h), value=pad_token)


def select_three_demos(demos: List[dict], random_demos: bool = False) -> List[dict]:
    if len(demos) == 0:
        raise ValueError("Task has no training demonstrations")
    if random_demos:
        import random as _rand
        if len(demos) >= 3:
            return _rand.sample(demos, 3)
        out = list(demos)
        while len(out) < 3:
            out.append(out[-1])
        return out
    out = list(demos[:3])
    while len(out) < 3:
        out.append(out[-1])
    return out


def build_tensors_from_json(task_path: Path, max_size: int, pad_token: int, random_demos: bool):
    with open(task_path) as f:
        task = json.load(f)
    demos = select_three_demos(task.get("train", []), random_demos=random_demos)
    demo_inputs = []
    demo_outputs = []
    for d in demos:
        inp = torch.tensor(d["input"], dtype=torch.long)
        out = torch.tensor(d["output"], dtype=torch.long)
        demo_inputs.append(pad_grid(inp, max_size=max_size, pad_token=pad_token))
        demo_outputs.append(pad_grid(out, max_size=max_size, pad_token=pad_token))
    demo_input = torch.stack(demo_inputs, dim=0).unsqueeze(0)   # (1,3,H,W)
    demo_output = torch.stack(demo_outputs, dim=0).unsqueeze(0) # (1,3,H,W)
    return demo_input, demo_output


def load_model_and_proj(ckpt_path: Path, hparams_path: Path, device: str):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    embed_dim = int(ckpt.get('embed_dim'))
    centroid_dim = int(ckpt.get('centroid_dim'))
    cat_order = ckpt.get('category_order', DEFAULT_CATS)

    # Load hparams to reconstruct architecture exactly
    with open(hparams_path) as f:
        hp = json.load(f)
    width_mult = float(hp.get('width_mult', 1.0))
    depth = int(hp.get('depth', 3))
    mlp_hidden = int(hp.get('mlp_hidden', 512))
    demo_agg = hp.get('demo_agg', 'flatten')
    use_coords = bool(hp.get('use_coords', False))

    model = TaskEncoderCNN(
        embed_dim=embed_dim,
        num_demos=3,
        width_mult=width_mult,
        depth=depth,
        mlp_hidden=mlp_hidden,
        demo_agg=demo_agg,
        use_coords=use_coords,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Centroid projection
    centroid_proj = None
    rand_proj = None
    if 'centroid_proj_state_dict' in ckpt:
        centroid_proj = nn.Linear(centroid_dim, embed_dim, bias=False).to(device)
        centroid_proj.load_state_dict(ckpt['centroid_proj_state_dict'])
        centroid_proj.eval()
    elif 'rand_proj' in ckpt:
        rand_proj = ckpt['rand_proj'].to(device)
    else:
        # Fallback to identity if dims match; else raise
        if centroid_dim != embed_dim:
            raise RuntimeError("No centroid projection found in checkpoint and dims mismatch.")

    return model, centroid_proj, rand_proj, cat_order, embed_dim, centroid_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='outputs/visual_classifier/models/task_encoder_direct_v3_best.pt')
    parser.add_argument('--centroids', type=str, default='outputs/visual_classifier/category_centroids_v3.npy')
    parser.add_argument('--hparams', type=str, default='outputs/visual_classifier/best_trial_params.json')
    parser.add_argument('--output', type=str, default='outputs/visual_classifier/results/arc2_classify_seed.csv')
    parser.add_argument('--max-size', type=int, default=30)
    parser.add_argument('--pad-token', type=int, default=10)
    parser.add_argument('--random-demos', action='store_true')
    args = parser.parse_args()

    base = REPRO_DIR
    in_dir = (base / args.input_dir) if not Path(args.input_dir).is_absolute() else Path(args.input_dir)
    ckpt_path = (base / args.checkpoint) if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    centroids_path = (base / args.centroids) if not Path(args.centroids).is_absolute() else Path(args.centroids)
    hparams_path = (base / args.hparams) if not Path(args.hparams).is_absolute() else Path(args.hparams)
    out_path = (base / args.output) if not Path(args.output).is_absolute() else Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load model and projection
    model, centroid_proj, rand_proj, cat_order, embed_dim, centroid_dim = load_model_and_proj(ckpt_path, hparams_path, device)

    # Load centroids (C, D)
    cent_np = np.load(str(centroids_path))
    centroids = torch.tensor(cent_np, dtype=torch.float32, device=device)
    if centroids.shape[1] != centroid_dim:
        raise RuntimeError(f"Centroid dim mismatch: file {centroids.shape[1]} vs ckpt {centroid_dim}")

    # Precompute projected centroids (C, E)
    if centroid_proj is not None:
        proj_centroids = centroid_proj(centroids)
    elif rand_proj is not None:
        proj_centroids = centroids @ rand_proj
    else:
        proj_centroids = centroids  # identity case when dims equal

    # Iterate tasks
    files = sorted(in_dir.glob('*.json'))
    if len(files) == 0:
        print(f"No JSON files found in {in_dir}")
        return

    header = ['task_id', 'pred_label', 'pred_idx'] + [f'prob_{c}' for c in cat_order]
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for fp in files:
            task_id = fp.stem
            demo_in, demo_out = build_tensors_from_json(fp, args.max_size, args.pad_token, args.random_demos)
            demo_in = demo_in.to(device)
            demo_out = demo_out.to(device)
            with torch.no_grad():
                emb = model(demo_in, demo_out)  # (1,E)
                logits = emb @ proj_centroids.t().contiguous()  # (1,C)
                probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
                pred_idx = int(np.argmax(probs))
                pred_label = cat_order[pred_idx]
            row = [task_id, pred_label, pred_idx] + [float(probs[i]) for i in range(len(cat_order))]
            writer.writerow(row)

    print(f"Saved results to: {out_path}")


if __name__ == '__main__':
    main()

