from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import shutil
try:
  from tqdm.auto import tqdm
except Exception:
  class tqdm:  # type: ignore
    def __init__(self, total=None, desc=None, leave=False):
      self.total = total
      self.n = 0
    def set_postfix(self, *_, **__):
      pass
    def update(self, n=1):
      self.n += n
    def close(self):
      pass


@dataclass
class SnapshotConfig:
  log_dir: Path
  run_prefix: str = "run"
  snapshot_nodes: int = 256
  snapshot_every_epochs: int = 1
  snapshot_quantiles: int = 3
  quantile_mode: str = "per_node"  # "per_node" | "global"
  node_mapping_path: Path | None = None
  labels_path: Path | None = None


class TGNExperimentLogger:
  def __init__(self, cfg: SnapshotConfig) -> None:
    self.cfg = cfg
    self.base = Path(cfg.log_dir) / cfg.run_prefix
    self.base.mkdir(parents=True, exist_ok=True)
    (self.base / "memory").mkdir(exist_ok=True)
    (self.base / "embeddings").mkdir(exist_ok=True)
    (self.base / "time_embeddings").mkdir(exist_ok=True)
    (self.base / "metrics").mkdir(exist_ok=True)
    self._materialise_node_mapping()
    self._materialise_labels()
    self._bar = None

  def _materialise_node_mapping(self) -> None:
    mapping_path = self.cfg.node_mapping_path
    if mapping_path is None:
      return
    source = Path(mapping_path)
    if not source.exists():
      return
    destination = self.base / source.name
    if destination.exists():
      return
    shutil.copy2(source, destination)

  def _materialise_labels(self) -> None:
    labels_path = self.cfg.labels_path
    if labels_path is None:
      return
    source = Path(labels_path)
    if not source.exists():
      return
    destination = self.base / source.name
    if destination.exists():
      return
    shutil.copy2(source, destination)

  # --- progress ---
  def start_epoch_bar(self, epoch: int, total_steps: int, desc: str | None = None) -> None:
    description = desc or f"Epoch {epoch}"
    self._bar = tqdm(total=total_steps, desc=description, leave=False)

  def update_epoch_bar(self, loss_value: float | None = None) -> None:
    if not self._bar:
      return
    if loss_value is not None:
      self._bar.set_postfix({"loss": f"{loss_value:.4f}"})
    self._bar.update(1)

  def close_epoch_bar(self) -> None:
    if self._bar:
      self._bar.close()
      self._bar = None

  # --- snapshots ---
  def snapshot_memory(self, tgn, epoch: int) -> None:  # tgn: model.tgn.TGN
    if not getattr(tgn, "use_memory", False) or not hasattr(tgn, "memory"):
      return
    mem = tgn.memory.memory.detach().cpu().numpy()
    last = tgn.memory.last_update.detach().cpu().numpy()
    np.savez(self.base / "memory" / f"epoch_{epoch:03d}.npz", memory=mem, last_update=last)

  def _select_customer_nodes(self, train_data, max_nodes: int) -> np.ndarray:
    # Use training sources as a proxy for customer nodes
    uniq = np.unique(train_data.sources)
    if len(uniq) <= max_nodes:
      return uniq
    rng = np.random.default_rng(0)
    return rng.choice(uniq, size=max_nodes, replace=False)

  def _pick_node_timepoints(
      self,
      node_id: int,
      train_data,
      *,
      K: int,
      quantile_mode: str,
      global_ts: np.ndarray | None = None,
  ) -> np.ndarray:
    """Return K timepoints for this node using per-node or global quantiles."""
    q = np.linspace(0.0, 1.0, K)
    if quantile_mode == "global":
      base_ts = np.asarray(global_ts if global_ts is not None else train_data.timestamps)
      return np.quantile(base_ts, q).astype(np.float32)

    # per_node mode
    mask = (train_data.sources == node_id)
    ts = train_data.timestamps[mask]
    if len(ts) == 0:
      base_ts = np.asarray(global_ts if global_ts is not None else train_data.timestamps)
      return np.quantile(base_ts, q).astype(np.float32)
    return np.quantile(np.sort(ts), q).astype(np.float32)

  @torch.no_grad()
  def snapshot_embeddings(
      self,
      tgn,                       # model.tgn.TGN
      train_data,                # utils.data_processing.Data
      epoch: int,
      n_neighbors: int,
      max_nodes: int | None = None,
      *,
      K: int | None = None,
      quantile_mode: str | None = None,
      global_ts: np.ndarray | None = None,
  ) -> None:
    max_nodes = max_nodes or self.cfg.snapshot_nodes
    nodes = self._select_customer_nodes(train_data, max_nodes)
    K = K or self.cfg.snapshot_quantiles
    quantile_mode = quantile_mode or self.cfg.quantile_mode
    times_list: List[np.ndarray] = [
      self._pick_node_timepoints(n, train_data, K=K, quantile_mode=quantile_mode, global_ts=global_ts)
      for n in nodes
    ]

    # Build batched query arrays (repeat each node for its 3 timepoints)
    reps = K
    sources = np.repeat(nodes, reps)
    timestamps = np.concatenate(times_list, axis=0).astype(np.float32)
    destinations = sources.copy()
    negatives = sources.copy()

    with tgn._no_memory_side_effects():
      src_e, _, _ = tgn.compute_temporal_embeddings(
        sources, destinations, negatives, timestamps, edge_idxs=None, n_neighbors=n_neighbors
      )

    emb = src_e.detach().cpu().numpy().reshape(len(nodes), reps, -1)
    # Also capture time encodings at these deltas
    if getattr(tgn, "use_memory", False):
      lu = tgn.memory.last_update[sources].detach().cpu().numpy()
      delta = torch.from_numpy(timestamps - lu.astype(np.float32)).to(tgn.device)
    else:
      delta = torch.from_numpy(timestamps).to(tgn.device)
    time_enc = tgn.time_encoder(delta.unsqueeze(1)).detach().cpu().numpy().reshape(len(nodes), reps, -1)

    out_path = self.base / "embeddings" / f"epoch_{epoch:03d}.npz"
    np.savez(out_path, node_ids=nodes, timepoints=np.stack(times_list), embeddings=emb)

    out_time_path = self.base / "time_embeddings" / f"epoch_{epoch:03d}.npz"
    np.savez(out_time_path, node_ids=nodes, timepoints=np.stack(times_list), time_embeddings=time_enc)
