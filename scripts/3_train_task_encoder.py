import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Sampler
import math

# Make reproduction/src importable as 'src'
REPRO_DIR = Path(__file__).resolve().parent.parent
if str(REPRO_DIR) not in sys.path:
    sys.path.insert(0, str(REPRO_DIR))

from src.models.task_encoder_cnn import TaskEncoderCNN  # noqa: E402
from src.models.task_encoder_advanced import TaskEncoderAdvanced  # noqa: E402
from src.data.arc_task_dataset import ARCTaskDataset, collate_arc_tasks  # noqa: E402


def resolve_labels_path(default_rel: str) -> Path:
    provided = Path(default_rel)
    if provided.is_absolute() and provided.exists():
        return provided
    path = provided.resolve()
    if path.exists():
        return path
    sp = Path(__file__).resolve()
    try:
        arc_reactor_dir = sp.parents[4]
        fallback = arc_reactor_dir / 'data' / 'taxonomy_classification' / 'all_tasks_classified.json'
        return fallback
    except IndexError:
        return path


essential_categories = ["S1","S2","S3","C1","C2","K1","L1","A1","A2"]


def make_splits(files, val_ratio=0.2, seed=42):
    files = list(files)
    random.Random(seed).shuffle(files)
    n_val = int(len(files) * val_ratio)
    return files[n_val:], files[:n_val]


def make_stratified_splits(files, labels_path: Path, category_to_idx: dict, val_ratio=0.2, seed=42):
    with open(labels_path) as f:
        task_categories = json.load(f)
    buckets = {i: [] for i in category_to_idx.values()}
    for fp in files:
        tid = Path(fp).stem
        cat = task_categories.get(tid, None)
        if cat is None or cat == 'ambiguous':
            continue
        if cat not in category_to_idx:
            continue
        buckets[category_to_idx[cat]].append(fp)
    train, val = [], []
    for k, lst in buckets.items():
        rng = random.Random(seed + k)
        rng.shuffle(lst)
        n_val = int(len(lst) * val_ratio)
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])
    return train, val


class BalancedBatchSampler(Sampler):
    """Yield balanced batches across categories (with replacement if needed).

    Ensures each batch contains roughly uniform samples per category. If batch_size < num_classes
    or not divisible, remainder is distributed across random categories.
    """
    def __init__(self, dataset, batch_size: int, num_classes: int, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.rng = random.Random(seed)
        # Build index buckets per class
        self.class_to_indices = {i: [] for i in range(num_classes)}
        for idx, (_, cidx) in enumerate(dataset.examples):
            self.class_to_indices[cidx].append(idx)
        # Fallback: if any class has no indices, keep list empty; sampling will still work with replacement safeguards
        self.length = max(1, math.ceil(len(self.dataset) / max(1, self.batch_size)))

    def __iter__(self):
        for _ in range(self.length):
            batch = []
            base = self.batch_size // self.num_classes
            rem = self.batch_size - base * self.num_classes
            # Sample base count per class
            for c in range(self.num_classes):
                pool = self.class_to_indices.get(c, [])
                for _ in range(base):
                    if pool:
                        batch.append(self.rng.choice(pool))
                    else:
                        # If no samples for this class, sample from any available index
                        fallback_idx = self.rng.randrange(len(self.dataset))
                        batch.append(fallback_idx)
            # Distribute remainder
            if rem > 0:
                cats = list(range(self.num_classes))
                self.rng.shuffle(cats)
                cats = cats[:rem]
                for c in cats:
                    pool = self.class_to_indices.get(c, [])
                    if pool:
                        batch.append(self.rng.choice(pool))
                    else:
                        batch.append(self.rng.randrange(len(self.dataset)))
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.length


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/distributional_alignment')
    parser.add_argument('--labels', type=str, default='../../../../data/taxonomy_classification/all_tasks_classified.json')
    parser.add_argument('--centroids', type=str, default='outputs/visual_classifier/category_centroids_v3.npy')
    parser.add_argument('--output-dir', type=str, default='../outputs/visual_classifier')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--encoder', type=str, default='cnn', choices=['cnn', 'context'])
    parser.add_argument('--class-weights', action='store_true')
    parser.add_argument('--use-cosine', action='store_true')
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--proj', type=str, default='learned', choices=['learned', 'random'])
    parser.add_argument('--use-scheduler', action='store_true')
    parser.add_argument('--early-stop-patience', type=int, default=0)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--use-sampler', action='store_true')
    parser.add_argument('--stratify', action='store_true')
    parser.add_argument('--random-demos', action='store_true')
    parser.add_argument('--color-permute', action='store_true')
    parser.add_argument('--log-confusion', action='store_true')
    # Model size controls
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--width-mult', type=float, default=1.0)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--mlp-hidden', type=int, default=512)
    parser.add_argument('--demo-agg', type=str, default='flatten', choices=['flatten', 'mean'])
    parser.add_argument('--use-coords', action='store_true')
    # Prototypes controls
    parser.add_argument('--use-learnable-prototypes', action='store_true')
    parser.add_argument('--prototypes-per-class', type=int, default=2)
    parser.add_argument('--proto-pool', type=str, default='max', choices=['max', 'mean', 'logsumexp'])
    parser.add_argument('--proto-init-noise', type=float, default=0.01)
    parser.add_argument('--proto-init-from-ckpt', type=str, default='')
    parser.add_argument('--proto-lr', type=float, default=5e-4)
    parser.add_argument('--proto-anchor-lambda', type=float, default=0.1)
    parser.add_argument('--freeze-encoder-epochs', type=int, default=0)
    # Batching
    parser.add_argument('--balanced-batches', action='store_true')
    # Linear head and center loss (alternative head to prototypes/centroids)
    parser.add_argument('--use-linear-head', action='store_true')
    parser.add_argument('--center-loss-lambda', type=float, default=0.0)
    parser.add_argument('--center-lr', type=float, default=5e-4)
    # ContextEncoder hyperparameters (only used if --encoder context)
    parser.add_argument('--ctx-d-model', type=int, default=256)
    parser.add_argument('--ctx-n-head', type=int, default=8)
    parser.add_argument('--ctx-pixel-layers', type=int, default=3)
    parser.add_argument('--ctx-grid-layers', type=int, default=2)
    parser.add_argument('--ctx-dropout', type=float, default=0.1)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    # Deterministic seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    data_dir = (REPRO_DIR / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    labels_path = resolve_labels_path(args.labels)
    centroids_path = (REPRO_DIR / args.centroids).resolve() if not Path(args.centroids).is_absolute() else Path(args.centroids)
    if not centroids_path.exists():
        # Fallback to reproduction/outputs location explicitly
        fallback_centroids = (REPRO_DIR / 'outputs' / 'visual_classifier' / 'category_centroids_v3.npy').resolve()
        if fallback_centroids.exists():
            centroids_path = fallback_centroids
    out_dir = (REPRO_DIR / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    (out_dir / 'models').mkdir(parents=True, exist_ok=True)
    (out_dir / 'results').mkdir(parents=True, exist_ok=True)

    task_files = sorted(list(data_dir.glob('*.json')))

    cat_to_idx_map = {name: i for i, name in enumerate(essential_categories)}
    if args.stratify:
        train_files, val_files = make_stratified_splits(task_files, labels_path, cat_to_idx_map, val_ratio=args.val_ratio)
    else:
        train_files, val_files = make_splits(task_files, val_ratio=args.val_ratio)
    train_ds = ARCTaskDataset(train_files, labels_path, max_grid_size=30, random_demos=args.random_demos, color_permute=args.color_permute)
    val_ds = ARCTaskDataset(val_files, labels_path, max_grid_size=30, random_demos=False)

    if args.balanced_batches:
        batch_sampler = BalancedBatchSampler(train_ds, batch_size=args.batch_size, num_classes=len(essential_categories))
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, collate_fn=collate_arc_tasks)
    elif args.use_sampler:
        counts = torch.zeros(len(essential_categories), dtype=torch.float32)
        for _, cidx in train_ds.examples:
            counts[cidx] += 1.0
        counts = torch.clamp(counts, min=1.0)
        class_weights = 1.0 / counts
        sample_weights = [class_weights[cidx].item() for _, cidx in train_ds.examples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_arc_tasks)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_arc_tasks)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_arc_tasks)

    # Load centroids (9, D)
    centroids_np = np.load(str(centroids_path))
    centroids = torch.tensor(centroids_np, dtype=torch.float32, device=device)
    num_categories, centroid_dim = centroids.shape

    # Model and (optional) projection/prototypes
    embed_dim = args.embed_dim
    if args.encoder == 'context':
        model = TaskEncoderAdvanced(
            embed_dim=embed_dim,
            num_demos=3,
            context_d_model=args.ctx_d_model,
            n_head=args.ctx_n_head,
            pixel_layers=args.ctx_pixel_layers,
            grid_layers=args.ctx_grid_layers,
            dropout_rate=args.ctx_dropout,
        ).to(device)
    else:
        model = TaskEncoderCNN(
            embed_dim=embed_dim,
            num_demos=3,
            width_mult=args.width_mult,
            depth=args.depth,
            mlp_hidden=args.mlp_hidden,
            demo_agg=args.demo_agg,
            use_coords=args.use_coords,
        ).to(device)
    linear_head = None
    centers = None
    prototypes = None
    centroid_proj = None
    rand_proj = None
    proj_params: list = []
    if args.use_linear_head:
        linear_head = nn.Linear(embed_dim, len(essential_categories)).to(device)
        if args.center_loss_lambda > 0:
            centers = nn.Parameter(torch.zeros(len(essential_categories), embed_dim, device=device))
    elif args.use_learnable_prototypes:
        # Initialize learnable prototypes from a random projection of centroids + noise
        torch.manual_seed(42)
        init_emb = None
        if args.proto_init_from_ckpt:
            ckpt_path = (REPRO_DIR / args.proto_init_from_ckpt) if not Path(args.proto_init_from_ckpt).is_absolute() else Path(args.proto_init_from_ckpt)
            if ckpt_path.exists():
                try:
                    ckpt = torch.load(str(ckpt_path), map_location=device)
                    state = ckpt.get('centroid_proj_state_dict')
                    if state is not None:
                        tmp_proj = nn.Linear(centroid_dim, embed_dim, bias=False).to(device)
                        tmp_proj.load_state_dict(state)
                        with torch.no_grad():
                            init_emb = tmp_proj(centroids)
                except Exception:
                    init_emb = None
        if init_emb is None:
            rand_proj = torch.randn(centroid_dim, embed_dim, device=device) / (centroid_dim ** 0.5)
            init_emb = centroids @ rand_proj  # (9, E)
        k = max(1, int(args.prototypes_per_class))
        prototypes = nn.Parameter(torch.empty(len(essential_categories), k, embed_dim, device=device))
        with torch.no_grad():
            for i in range(k):
                noise = args.proto_init_noise
                prototypes[:, i, :] = init_emb + noise * torch.randn_like(init_emb)
        proto_anchor_ref = init_emb.clone().detach()  # (C,E)
    else:
        # Use fixed centroids with projection (learned or random)
        if args.proj == 'learned':
            centroid_proj = nn.Linear(centroid_dim, embed_dim, bias=False).to(device)
            proj_params = list(centroid_proj.parameters())
            rand_proj = None
        else:
            centroid_proj = None
            torch.manual_seed(42)
            rand_proj = torch.randn(centroid_dim, embed_dim, device=device) / (centroid_dim ** 0.5)
            proj_params = []

    # Optimizer param groups
    param_groups = [
        {'params': list(model.parameters()), 'lr': args.lr, 'weight_decay': args.weight_decay}
    ]
    if len(proj_params) > 0:
        param_groups.append({'params': proj_params, 'lr': args.lr, 'weight_decay': args.weight_decay})
    if prototypes is not None:
        param_groups.append({'params': [prototypes], 'lr': args.proto_lr, 'weight_decay': 0.0})
    if linear_head is not None:
        param_groups.append({'params': linear_head.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
    if centers is not None:
        param_groups.append({'params': [centers], 'lr': args.center_lr, 'weight_decay': 0.0})
    optimizer = torch.optim.AdamW(param_groups)

    if args.class_weights:
        # Compute class weights from training distribution
        counts = torch.zeros(len(essential_categories), dtype=torch.float32)
        for _, cidx in train_ds.examples:
            counts[cidx] += 1.0
        counts = torch.clamp(counts, min=1.0)
        inv_freq = 1.0 / counts
        weights = inv_freq / inv_freq.sum() * len(essential_categories)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-5)
    else:
        scheduler = None

    best_val_acc = 0.0
    best_ckpt = out_dir / 'models' / 'task_encoder_direct_v3_best.pt'
    metrics_path = out_dir / 'results' / 'phase1_training_metrics_v3.json'
    metrics = {"epochs": []}

    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if centroid_proj is not None:
            centroid_proj.train()
        # Optional encoder freeze for first N epochs (to let prototypes adapt)
        if args.freeze_encoder_epochs > 0:
            if epoch == 1:
                model.requires_grad_(False)
            if epoch == args.freeze_encoder_epochs + 1:
                model.requires_grad_(True)
        train_loss = 0.0
        train_acc = 0.0
        steps = 0

        for demo_in, demo_out, cat_idx in train_loader:
            demo_in = demo_in.to(device)
            demo_out = demo_out.to(device)
            cat_idx = cat_idx.to(device)
            optimizer.zero_grad()
            emb = model(demo_in, demo_out)
            if linear_head is not None:
                logits = linear_head(emb)
            elif prototypes is not None:
                # Similarity to prototypes: (B,E) vs (C,K,E) -> (B,C,K)
                if args.use_cosine:
                    emb_n = F.normalize(emb, dim=-1)
                    prot_n = F.normalize(prototypes, dim=-1)
                    sim = torch.einsum('be,cke->bck', emb_n, prot_n)
                    sim = sim * args.temperature
                else:
                    sim = torch.einsum('be,cke->bck', emb, prototypes)
                if args.proto_pool == 'max':
                    logits = sim.max(dim=2).values
                elif args.proto_pool == 'mean':
                    logits = sim.mean(dim=2)
                else:  # logsumexp
                    logits = torch.logsumexp(sim, dim=2)
            else:
                # Fixed centroids path
                if centroid_proj is not None:
                    proj_centroids = centroid_proj(centroids)
                else:
                    proj_centroids = centroids @ rand_proj
                if args.use_cosine:
                    emb_n = F.normalize(emb, dim=-1)
                    cent_n = F.normalize(proj_centroids, dim=-1)
                    logits = emb_n @ cent_n.t().contiguous()
                    logits = logits * args.temperature
                else:
                    logits = emb @ proj_centroids.t().contiguous()
            loss = criterion(logits, cat_idx)
            # Optional center loss encourages tighter clusters toward learned centers
            if centers is not None and args.center_loss_lambda > 0:
                tgt_centers = centers[cat_idx]
                loss = loss + args.center_loss_lambda * F.mse_loss(emb, tgt_centers)
            # Prototype anchor regularization toward init embeddings
            if prototypes is not None and args.proto_anchor_lambda > 0:
                anchor_target = proto_anchor_ref.unsqueeze(1).expand_as(prototypes)
                anchor_loss = F.mse_loss(prototypes, anchor_target)
                loss = loss + args.proto_anchor_lambda * anchor_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy_from_logits(logits, cat_idx)
            steps += 1

        train_loss /= max(1, steps)
        train_acc /= max(1, steps)

        # Validation
        model.eval()
        if centroid_proj is not None:
            centroid_proj.eval()
        val_loss = 0.0
        val_acc = 0.0
        vsteps = 0
        with torch.no_grad():
            # Per-category counters
            correct_by_cat = torch.zeros(len(essential_categories), dtype=torch.long)
            total_by_cat = torch.zeros(len(essential_categories), dtype=torch.long)
            if args.log_confusion:
                confusion = torch.zeros(len(essential_categories), len(essential_categories), dtype=torch.long)
            for demo_in, demo_out, cat_idx in val_loader:
                demo_in = demo_in.to(device)
                demo_out = demo_out.to(device)
                cat_idx = cat_idx.to(device)
                emb = model(demo_in, demo_out)
                if linear_head is not None:
                    logits = linear_head(emb)
                elif prototypes is not None:
                    if args.use_cosine:
                        emb_n = F.normalize(emb, dim=-1)
                        prot_n = F.normalize(prototypes, dim=-1)
                        sim = torch.einsum('be,cke->bck', emb_n, prot_n)
                        sim = sim * args.temperature
                    else:
                        sim = torch.einsum('be,cke->bck', emb, prototypes)
                    if args.proto_pool == 'max':
                        logits = sim.max(dim=2).values
                    elif args.proto_pool == 'mean':
                        logits = sim.mean(dim=2)
                    else:
                        logits = torch.logsumexp(sim, dim=2)
                else:
                    if centroid_proj is not None:
                        proj_centroids = centroid_proj(centroids)
                    else:
                        proj_centroids = centroids @ rand_proj
                    if args.use_cosine:
                        emb_n = F.normalize(emb, dim=-1)
                        cent_n = F.normalize(proj_centroids, dim=-1)
                        logits = emb_n @ cent_n.t().contiguous()
                        logits = logits * args.temperature
                    else:
                        logits = emb @ proj_centroids.t().contiguous()
                loss = criterion(logits, cat_idx)
                val_loss += loss.item()
                val_acc += accuracy_from_logits(logits, cat_idx)
                vsteps += 1
                preds = torch.argmax(logits, dim=-1)
                for k in range(len(essential_categories)):
                    mask = (cat_idx == k)
                    total_by_cat[k] += mask.sum().item()
                    correct_by_cat[k] += (preds[mask] == k).sum().item()
                if args.log_confusion:
                    for t, p in zip(cat_idx.cpu().tolist(), preds.cpu().tolist()):
                        confusion[t][p] += 1
        val_loss /= max(1, vsteps)
        val_acc /= max(1, vsteps)
        if scheduler is not None:
            scheduler.step(val_loss)

        # Compute per-category accuracy
        val_cat_acc = {}
        for i, name in enumerate(essential_categories):
            tot = int(total_by_cat[i].item())
            cor = int(correct_by_cat[i].item())
            val_cat_acc[name] = (cor / tot) if tot > 0 else None

        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_cat_acc": val_cat_acc,
        }
        if args.log_confusion:
            rec["val_confusion"] = confusion.tolist()
        metrics["epochs"].append(rec)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                'model_state_dict': model.state_dict(),
                'embed_dim': embed_dim,
                'centroid_dim': centroid_dim,
                'category_order': essential_categories,
            }
            if centroid_proj is not None:
                ckpt['centroid_proj_state_dict'] = centroid_proj.state_dict()
            elif rand_proj is not None:
                ckpt['rand_proj'] = rand_proj.detach().cpu()
            if prototypes is not None:
                ckpt['prototypes'] = prototypes.detach().cpu()
            torch.save(ckpt, best_ckpt)
            no_improve = 0
        else:
            no_improve += 1

        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            break

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Best val_acc={best_val_acc:.3f}")
    print(f"Saved best checkpoint to: {best_ckpt}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == '__main__':
    main()
