from __future__ import annotations

import argparse
import copy
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

from load_dataset import load_dataset
from models import GCNLinkPredictor, GCNNodeClassifier, LargeLinearNodeClassifier


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hits_at_k(pos_scores: torch.Tensor, neg_scores: torch.Tensor, k: int = 50) -> float:
    n_neg_higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
    return (n_neg_higher < k).float().mean().item()


def save_model(model: torch.nn.Module, path: Path) -> None:
    if hasattr(model, "reset_runtime_state"):
        model.reset_runtime_state()
    model_cpu = copy.deepcopy(model).cpu()
    if hasattr(model_cpu, "reset_runtime_state"):
        model_cpu.reset_runtime_state()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_cpu, path)


def choose_device(preference: str) -> torch.device:
    def _cuda_supported() -> bool:
        if not torch.cuda.is_available():
            return False

        try:
            capability = torch.cuda.get_device_capability()
            arch = f"sm_{capability[0]}{capability[1]}"
            supported_arches = set(torch.cuda.get_arch_list())
            if supported_arches and arch not in supported_arches:
                print(
                    f"CUDA device architecture {arch} is not supported by this PyTorch build; "
                    "falling back to CPU.",
                    flush=True,
                )
                return False

            probe = torch.zeros(1, device="cuda")
            probe.add_(1.0)
            return True
        except Exception as exc:
            print(
                f"CUDA initialization failed ({exc}); falling back to CPU.",
                flush=True,
            )
            return False

    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not _cuda_supported():
            raise RuntimeError("CUDA requested but not usable with this PyTorch installation")
        return torch.device("cuda")
    return torch.device("cuda" if _cuda_supported() else "cpu")


@dataclass
class TrainResult:
    model: torch.nn.Module
    best_metric: float
    metric_name: str
    elapsed_sec: float


def resolve_time_budget_seconds(dataset: str, override_seconds: float | None) -> float:
    default_budget = {"A": 3600.0, "B": 7200.0, "C": 7200.0}
    if override_seconds is not None:
        return float(override_seconds)
    return default_budget[dataset]


def compute_safety_margin_seconds(time_budget_seconds: float) -> float:
    margin = min(60.0, max(5.0, 0.02 * float(time_budget_seconds)))
    return min(margin, max(0.5, 0.25 * float(time_budget_seconds)))


def deadline_reached(deadline: float | None, extra_buffer_seconds: float = 0.0) -> bool:
    return deadline is not None and (time.perf_counter() + extra_buffer_seconds) >= deadline


def train_a(
    data_dir: str,
    device: torch.device,
    max_epochs: int,
    patience: int,
    deadline: float | None,
) -> TrainResult:
    dataset = load_dataset("A", data_dir)
    data = dataset[0]
    data = data.to(device)

    train_idx = data.labeled_nodes[data.train_mask]
    val_idx = data.labeled_nodes[data.val_mask]
    y_train = data.y[data.train_mask]
    y_val = data.y[data.val_mask]

    model = GCNNodeClassifier(
        input_dim=data.x.size(1),
        hidden_dim=64,
        output_dim=dataset.num_classes,
        num_layers=2,
        dropout=0.5,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_metric = -1.0
    best_state = None
    stale_epochs = 0
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        if deadline_reached(deadline):
            print("[A] stopping before next epoch due to time budget", flush=True)
            break

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_idx], y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            pred = logits[val_idx].argmax(dim=1)
            val_acc = accuracy_score(y_val.cpu().numpy(), pred.cpu().numpy())

        print(
            f"[A] epoch={epoch:03d} loss={loss.item():.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_metric:
            best_metric = float(val_acc)
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_metric = float("nan")

    model.load_state_dict(best_state)
    return TrainResult(
        model=model,
        best_metric=best_metric,
        metric_name="val_acc",
        elapsed_sec=time.perf_counter() - start_time,
    )


def resolve_node_data_file(data_dir: str, dataset_name: str) -> str:
    for filename in ("data.pt", "test.pt"):
        path = os.path.join(data_dir, dataset_name, filename)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No data.pt or test.pt found for dataset {dataset_name} in {data_dir}")


def iter_index_batches(index: torch.Tensor, batch_size: int, shuffle: bool) -> torch.Tensor:
    if shuffle:
        order = torch.randperm(index.numel())
        index = index[order]
    for start in range(0, index.numel(), batch_size):
        yield index[start:start + batch_size]


def batched_linear_scores(
    model: LargeLinearNodeClassifier,
    x: torch.Tensor,
    node_index: torch.Tensor,
    degree_feature: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch_nodes in iter_index_batches(node_index, batch_size=batch_size, shuffle=False):
            x_batch = x[batch_nodes].to(device)
            deg_batch = None if degree_feature is None else degree_feature[batch_nodes].to(device)
            logits = model.forward_batch(x_batch, deg_batch).squeeze(-1)
            outputs.append(logits.cpu())
    return torch.cat(outputs, dim=0)


def train_b(
    data_dir: str,
    device: torch.device,
    max_epochs: int,
    patience: int,
    deadline: float | None,
) -> TrainResult:
    data_path = resolve_node_data_file(data_dir, "B")
    data = torch.load(data_path, weights_only=False, mmap=True, map_location="cpu")

    train_nodes = data.labeled_nodes[data.train_mask]
    val_nodes = data.labeled_nodes[data.val_mask]
    y_train = data.y[data.train_mask].float()
    y_val = data.y[data.val_mask].float()

    train_pos = float(y_train.sum().item())
    train_neg = float(y_train.numel() - train_pos)
    pos_weight = torch.tensor([max(1.0, train_neg / max(train_pos, 1.0))], device=device)

    degree_feature = None
    model = LargeLinearNodeClassifier(
        input_dim=data.x.size(1),
        use_degree_feature=True,
        inference_chunk_size=131072,
    ).to(device)

    prior = max(1e-6, min(1.0 - 1e-6, float(y_train.mean().item())))
    model.linear.bias.data.fill_(float(np.log(prior / (1.0 - prior))))

    degree_feature = model._degree_feature(data.edge_index, data.x.size(0), torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    batch_size = 32768 if device.type == "cpu" else 16384
    eval_batch_size = 65536 if device.type == "cpu" else 32768
    best_metric = -1.0
    best_state = None
    stale_epochs = 0
    start_time = time.perf_counter()
    train_positions = torch.arange(train_nodes.numel())

    for epoch in range(1, max_epochs + 1):
        if deadline_reached(deadline):
            print("[B] stopping before next epoch due to time budget", flush=True)
            break

        model.train()
        epoch_loss = 0.0
        seen = 0
        interrupted_for_time = False

        for batch_pos in iter_index_batches(train_positions, batch_size=batch_size, shuffle=True):
            if deadline_reached(deadline, extra_buffer_seconds=1.0) and best_state is not None:
                interrupted_for_time = True
                print(f"[B] stopping during epoch {epoch:02d} due to time budget", flush=True)
                break

            batch_nodes = train_nodes[batch_pos]
            x_batch = data.x[batch_nodes].to(device)
            deg_batch = degree_feature[batch_nodes].to(device)
            y_batch = y_train[batch_pos].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_batch(x_batch, deg_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_nodes.numel()
            epoch_loss += loss.item() * batch_size_actual
            seen += batch_size_actual

        if seen == 0:
            break

        if interrupted_for_time and best_state is not None:
            break

        val_scores = batched_linear_scores(
            model=model,
            x=data.x,
            node_index=val_nodes,
            degree_feature=degree_feature,
            batch_size=eval_batch_size,
            device=device,
        )
        val_auc = roc_auc_score(y_val.numpy(), val_scores.numpy())
        mean_loss = epoch_loss / max(seen, 1)
        print(
            f"[B] epoch={epoch:02d} loss={mean_loss:.6f} val_auc={val_auc:.6f}",
            flush=True,
        )

        if val_auc > best_metric:
            best_metric = float(val_auc)
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_metric = float("nan")

    model.load_state_dict(best_state)
    return TrainResult(
        model=model,
        best_metric=best_metric,
        metric_name="val_auc",
        elapsed_sec=time.perf_counter() - start_time,
    )


def train_c(
    data_dir: str,
    device: torch.device,
    max_epochs: int,
    patience: int,
    deadline: float | None,
) -> TrainResult:
    dataset = load_dataset("C", data_dir)

    x = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)
    train_pos = dataset.train_pos.to(device)
    train_neg = dataset.train_neg.to(device)
    valid_pos = dataset.valid_pos.to(device)
    valid_neg = dataset.valid_neg.to(device)

    model = GCNLinkPredictor(
        input_dim=x.size(1),
        hidden_dim=256,
        embedding_dim=128,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    best_metric = -1.0
    best_state = None
    stale_epochs = 0
    start_time = time.perf_counter()

    target_pos = torch.ones(train_pos.size(0), device=device)
    target_neg = torch.zeros(train_neg.size(0), device=device)

    for epoch in range(1, max_epochs + 1):
        if deadline_reached(deadline):
            print("[C] stopping before next epoch due to time budget", flush=True)
            break

        model.train()
        optimizer.zero_grad(set_to_none=True)

        pos_scores = model(x, edge_index, train_pos)
        neg_scores = model(x, edge_index, train_neg)
        train_scores = torch.cat([pos_scores, neg_scores], dim=0)
        train_targets = torch.cat([target_pos, target_neg], dim=0)
        loss = F.binary_cross_entropy_with_logits(train_scores, train_targets)
        loss.backward()
        optimizer.step()

        model.eval()
        model.reset_runtime_state()
        with torch.no_grad():
            pos_scores = model(x, edge_index, valid_pos)
            neg_scores = model(x, edge_index, valid_neg.view(-1, 2)).view(valid_neg.size(0), valid_neg.size(1))
            val_hits = hits_at_k(pos_scores, neg_scores, k=50)

        print(
            f"[C] epoch={epoch:03d} loss={loss.item():.4f} val_hits@50={val_hits:.4f}",
            flush=True,
        )

        if val_hits > best_metric:
            best_metric = float(val_hits)
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_metric = float("nan")

    model.load_state_dict(best_state)
    return TrainResult(
        model=model,
        best_metric=best_metric,
        metric_name="val_hits@50",
        elapsed_sec=time.perf_counter() - start_time,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train COL761 A3 graph models.")
    parser.add_argument("--dataset", required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", required=True, choices=["node", "link"])
    parser.add_argument("--data_dir", required=True, help="Absolute path to datasets")
    parser.add_argument("--model_dir", required=True, help="Directory where model will be saved")
    parser.add_argument("--kerberos", required=True, help="Kerberos ID used in output filename")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=761)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--time_budget_seconds",
        type=float,
        default=None,
        help="Optional wall-clock training budget in seconds. Defaults to 3600 for A and 7200 for B/C.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    valid = {"node": ("A", "B"), "link": ("C",)}
    if args.dataset not in valid[args.task]:
        raise ValueError(
            f"--task {args.task} is not valid for --dataset {args.dataset}. "
            f"Expected dataset in {valid[args.task]}."
        )

    seed_everything(args.seed)
    device = choose_device(args.device)

    default_epochs = {"A": 100000, "B": 100000, "C": 100000}
    default_patience = {"A": 100000, "B": 100000, "C": 100000}
    max_epochs = default_epochs[args.dataset] if args.max_epochs is None else args.max_epochs
    patience = default_patience[args.dataset] if args.patience is None else args.patience
    time_budget_seconds = resolve_time_budget_seconds(args.dataset, args.time_budget_seconds)
    safety_margin_seconds = compute_safety_margin_seconds(time_budget_seconds)
    deadline = time.perf_counter() + max(0.0, time_budget_seconds - safety_margin_seconds)

    print(f"Using device: {device}", flush=True)
    print(f"Training dataset {args.dataset} for task {args.task}", flush=True)
    print(
        f"Time budget: {time_budget_seconds:.1f}s with {safety_margin_seconds:.1f}s save margin",
        flush=True,
    )

    if args.dataset == "A":
        result = train_a(
            args.data_dir,
            device,
            max_epochs=max_epochs,
            patience=patience,
            deadline=deadline,
        )
    elif args.dataset == "B":
        result = train_b(
            args.data_dir,
            device,
            max_epochs=max_epochs,
            patience=patience,
            deadline=deadline,
        )
    else:
        result = train_c(
            args.data_dir,
            device,
            max_epochs=max_epochs,
            patience=patience,
            deadline=deadline,
        )

    model_path = Path(args.model_dir) / f"{args.kerberos}_model_{args.dataset}.pt"
    save_model(result.model, model_path)
    print(
        f"Saved model to {model_path} | {result.metric_name}={result.best_metric:.6f} | "
        f"elapsed={result.elapsed_sec:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
