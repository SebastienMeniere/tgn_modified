from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pickle

import numpy as np
import pandas as pd


def md5_of_file(path: Path) -> str | None:
  try:
    h = hashlib.md5()
    with open(path, 'rb') as f:
      for chunk in iter(lambda: f.read(8192), b''):
        h.update(chunk)
    return h.hexdigest()
  except FileNotFoundError:
    return None


def load_snapshot(log_dir: Path, run_prefix: str, epoch: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
  snap_path = log_dir / run_prefix / 'embeddings' / f'epoch_{epoch:03d}.npz'
  if not snap_path.exists():
    raise FileNotFoundError(f"Snapshot not found: {snap_path}")
  snap = np.load(snap_path)
  node_ids = snap['node_ids']
  embeddings = snap['embeddings']  # [N, K, D]
  timepoints = snap['timepoints']   # [N, K]
  return node_ids, embeddings, timepoints, snap_path


def load_labels(processed_dir: Path, run_dir: Path, dataset: str) -> Tuple[pd.DataFrame, Path]:
  # Prefer labels co-located with the run if present
  run_labels = run_dir / f'{dataset}_labels.csv'
  if run_labels.exists():
    labels_path = run_labels
  else:
    labels_path = processed_dir / f'{dataset}_labels.csv'
    if not labels_path.exists():
      # Backward-compat synthetic default
      labels_path = processed_dir / f'{dataset}/{dataset}_labels.csv'
  if not labels_path.exists():
    raise FileNotFoundError(f"Labels CSV not found near run or processed dir for dataset '{dataset}'")
  df = pd.read_csv(labels_path)
  required = {'node_id', 'node_key', 'customer_id', 'cluster'}
  missing = required - set(df.columns)
  if missing:
    raise ValueError(f"Labels file missing columns: {sorted(missing)} in {labels_path}")
  return df, labels_path


def verify_mapping_integrity(processed_dir: Path, run_dir: Path, dataset: str) -> Dict[str, object]:
  proc_map = processed_dir / f'{dataset}_node2id.pkl'
  run_map = run_dir / f'{dataset}_node2id.pkl'
  return {
    'processed_mapping_path': str(proc_map),
    'run_mapping_path': str(run_map),
    'processed_mapping_md5': md5_of_file(proc_map),
    'run_mapping_md5': md5_of_file(run_map),
    'run_has_mapping_copy': run_map.exists(),
    'processed_has_mapping': proc_map.exists(),
    'mappings_match': (md5_of_file(proc_map) == md5_of_file(run_map)) if (proc_map.exists() and run_map.exists()) else None,
  }


def fit_umap(X: np.ndarray, *, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
  try:
    import umap
  except Exception as e:
    raise RuntimeError("UMAP is not installed. Please install `umap-learn`.") from e
  reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
  return reducer.fit_transform(X)


def plot_scatter(out_path: Path, coords: np.ndarray, labels: pd.Series, title: str) -> None:
  import matplotlib.pyplot as plt
  plt.figure(figsize=(8, 6))
  uniq = sorted(labels.unique())
  for lab in uniq:
    mask = (labels == lab).values
    plt.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.8, label=str(lab))
  plt.legend(title='Cluster', frameon=True)
  plt.title(title)
  plt.tight_layout()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(out_path, dpi=150)
  plt.close()


def main() -> None:
  ap = argparse.ArgumentParser(description='Analyze TGN snapshot with label-safe pipeline (no drift).')
  ap.add_argument('--log-dir', type=Path, default=Path('experiment_logs'), help='Base logs directory')
  ap.add_argument('--run-prefix', type=str, required=True, help='Run prefix used during training')
  ap.add_argument('--epoch', type=int, default=0, help='Snapshot epoch index')
  ap.add_argument('--processed-dir', type=Path, default=Path('../data/processed/synthetic'), help='Processed dataset folder')
  ap.add_argument('--dataset', type=str, default='synthetic', help='Dataset name (prefix for files)')
  ap.add_argument('--slice', type=str, default='final', help="Time slice to plot: 'final' or integer index")
  ap.add_argument('--umap-n-neighbors', type=int, default=15)
  ap.add_argument('--umap-min-dist', type=float, default=0.1)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out-dir', type=Path, default=None, help='Optional custom output dir; defaults to run dir / plots')
  args = ap.parse_args()

  node_ids, embeddings, timepoints, snap_path = load_snapshot(args.log_dir, args.run_prefix, args.epoch)
  N, K, D = embeddings.shape
  run_dir = snap_path.parent.parent  # .../<run>/

  labels_df, labels_path = load_labels(args.processed_dir, run_dir, args.dataset)
  # Prefer robust name-based mapping via the run's node2id (numeric ids may differ between runs)
  name_to_cluster = labels_df.set_index('node_key')['cluster']
  run_map_path = run_dir / f'{args.dataset}_node2id.pkl'
  if run_map_path.exists():
    with open(run_map_path, 'rb') as f:
      node2id_run = pickle.load(f)
    id2node_run = {v: k for k, v in node2id_run.items()}
  else:
    # Fallback to processed mapping if run mapping missing
    proc_map_path = args.processed_dir / f'{args.dataset}_node2id.pkl'
    with open(proc_map_path, 'rb') as f:
      node2id_proc = pickle.load(f)
    id2node_run = {v: k for k, v in node2id_proc.items()}

  # Alignment checks
  # Build clusters by joining node_ids -> node_key (from run mapping) -> cluster name
  node_names = pd.Series(node_ids).map(id2node_run)
  clusters_by_name = node_names.map(name_to_cluster)
  unknown = int(clusters_by_name.isna().sum())
  cust_mask = clusters_by_name.notna().values
  cust_clusters = clusters_by_name[cust_mask].reset_index(drop=True)

  # Slice selection
  if args.slice == 'final':
    k_idx = K - 1
  else:
    k_idx = int(args.slice)
    if not (0 <= k_idx < K):
      raise ValueError(f"slice index {k_idx} out of range [0,{K-1}]")

  X = embeddings[cust_mask, k_idx, :]

  coords = fit_umap(X, n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist, seed=args.seed)

  # Per-cluster statistics in the original embedding space (before UMAP)
  cluster_centroids = {}
  cluster_norms = {}
  for label in sorted(pd.unique(cust_clusters).tolist()):
    mask_label = (cust_clusters == label).values
    vecs = X[mask_label]
    if len(vecs) == 0:
      continue
    centroid = vecs.mean(axis=0)
    cluster_centroids[label] = centroid.tolist()
    cluster_norms[label] = float(np.linalg.norm(centroid))

  pairwise_cosine: Dict[str, float] = {}
  labels_sorted = sorted(cluster_centroids.keys())
  for i in range(len(labels_sorted)):
    for j in range(i + 1, len(labels_sorted)):
      a = np.array(cluster_centroids[labels_sorted[i]])
      b = np.array(cluster_centroids[labels_sorted[j]])
      denom = (np.linalg.norm(a) * np.linalg.norm(b))
      if denom <= 0:
        cosine = float('nan')
      else:
        cosine = float(np.dot(a, b) / denom)
      pairwise_cosine[f"{labels_sorted[i]}__{labels_sorted[j]}"] = cosine

  # Paths & plotting
  out_dir = args.out_dir or (run_dir / 'plots')
  plot_path = out_dir / f"umap_{args.dataset}_epoch_{args.epoch:03d}_k{k_idx}.png"
  plot_scatter(plot_path, coords, cust_clusters, title=f"UMAP (epoch {args.epoch}, slice {k_idx})")

  # Integrity report
  map_info = verify_mapping_integrity(args.processed_dir, run_dir, args.dataset)
  report = {
    'snapshot_path': str(snap_path),
    'run_dir': str(run_dir),
    'dataset': args.dataset,
    'processed_dir': str(args.processed_dir),
    'labels_path_used': str(labels_path),
    'shape': {'N': int(N), 'K': int(K), 'D': int(D)},
    'slice_index': int(k_idx),
    'n_customers_in_snapshot': int(cust_mask.sum()),
    'unknown_labels': unknown,
    'unique_clusters': sorted(pd.unique(cust_clusters).tolist()),
    'cluster_counts': cust_clusters.value_counts().sort_index().to_dict(),
    'cluster_centroids': cluster_centroids,
    'cluster_centroid_norms': cluster_norms,
    'cluster_pairwise_cosine': pairwise_cosine,
    'mapping': map_info,
    'outputs': {
      'plot_path': str(plot_path),
    },
    'head_labels': labels_df.head(10).to_dict(orient='records'),
  }

  out_dir.mkdir(parents=True, exist_ok=True)
  report_path = out_dir / f"analysis_{args.dataset}_epoch_{args.epoch:03d}_k{k_idx}.json"
  with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

  # Console summary for quick inspection
  print("--- Snapshot analysis summary ---")
  print(f"snapshot: {snap_path}")
  print(f"nodes/emb dims: N={N}, K={K}, D={D}; slice={k_idx}")
  print(f"customers in snapshot: {int(cust_mask.sum())} (unknown labels: {unknown})")
  print(f"unique clusters: {sorted(pd.unique(cust_clusters).tolist())}")
  print(f"plot: {plot_path}")
  print(f"report: {report_path}")


if __name__ == '__main__':
  main()
