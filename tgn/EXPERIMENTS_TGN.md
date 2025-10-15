TGN Synthetic Retail Experiment Framework

Purpose
- Establish a clear, reproducible plan to evaluate what the TGN can model in a retail setting using the synthetic, parameterized data generator.
- Define questions, required data, concrete metrics, and where to hook into the code to capture evidence during training and evaluation.

Key Code Components
- Training loop: soakn/tgn/train_self_supervised.py:1
- Model and scoring helpers: soakn/tgn/model/tgn.py:1
- Temporal attention (returns attention weights): soakn/tgn/model/temporal_attention.py:1
- Embedding modules (graph attention/sum, identity, time): soakn/tgn/modules/embedding_module.py:1
- Memory module and updater (GRU/RNN): soakn/tgn/modules/memory.py:1, soakn/tgn/modules/memory_updater.py:1
- Evaluation utilities: soakn/tgn/evaluation/evaluation.py:1
- Data loader/splits: soakn/tgn/utils/data_processing.py:1

Data Inputs
- Synthetic specs: products.csv, cluster_probs.csv, clusters.csv
- Generated events: synthetic_data.csv (or preprocessed `synthetic_artifacts`)
- Preprocessed artifacts for TGN: synthetic.csv, synthetic.npy, synthetic_node.npy, synthetic_node2id.pkl

Terminology & Dimensions
- Node embeddings: latent embedding from `embedding_module`
- Memory: per-node memory vector when `--use_memory` is enabled
- Time encoding: feature from `TimeEncode`
- Message: raw message constructed for memory updates

Research Questions and Designs

1) Do customers with similar purchase behavior cluster together in embedding space?
- Statement: Customers drawn from distributions with similar product affinities should be close in the learned embedding space.
- Data:
  - Customer node embeddings at one or more reference times (e.g., first event, mid, last).
  - Ground-truth customer labels (cluster name) from the synthetic generation; accessible via event column `cluster` in synthetic_data.csv and preserved via preprocessing.
- Design:
  - Snapshot embeddings for all customers at consistent timepoints without side effects using `TGN.compute_temporal_embeddings` within `_no_memory_side_effects()` (soakn/tgn/model/tgn.py:91).
  - Compute clustering quality metrics: Silhouette, Calinski–Harabasz (unsupervised); ARI/NMI (supervised) using cluster labels.
  - Visualize with UMAP/t-SNE using fixed random seeds; color by cluster.
- Minimal instrumentation:
  - Add a helper that fetches customer embeddings at specified timestamps using `compute_temporal_embeddings` with dummy negatives (see `compute_edge_logits_many_negs_no_update` for tiling patterns; soakn/tgn/model/tgn.py:155).
  - Persist embeddings per customer and timepoint as a Parquet/NumPy file for offline analysis.

2) Can TGN capture distinct temporal patterns of transitions in buying behavior?
- Statement: Embedding trajectories over time should reflect the known decay/transition from start to end distributions (e.g., linear, exponential, parameterizable, sigmoid).
- Data:
  - Embedding trajectories: for each customer, embedding at each event timestamp (or selected subsampled times).
  - Ground-truth progress proxies: normalized event index; decay parameters α, τ from clusters.csv.
- Design:
  - For each customer, compute trajectory distance to start/end centroids; expect monotone movement toward end centroid with shape consistent to cluster decay.
  - Fit simple parametric curves to distance-to-start vs normalized time and compare with cluster’s decay family (e.g., R² against piecewise exponential/logistic templates).
  - Compute DTW between trajectories and idealized templates; report alignment cost per cluster.
- Instrumentation:
  - Same embedding snapshots as (1), stored per customer per event index with timestamp and cluster label.

3) Is the memory (and memory updater) interpretable during/after training?
- Statement: Memory vectors should encode recent behavioral context; question is whether we can extract structure.
- Data:
  - Per-epoch memory snapshots: `tgn.memory.memory` and `tgn.memory.last_update` (see `backup_memory()` in soakn/tgn/modules/memory.py:24).
  - Optionally, raw messages counts per node (via `memory.messages`).
- Design:
  - PCA/UMAP on memory vectors; color by cluster and by normalized progress.
  - Linear probe: predict current category affinities (from cluster_probs at the current normalized progress) from memory; report R² / correlation.
  - Track gradient norms for memory-related components (already logged by train loop; see soakn/tgn/train_self_supervised.py:294+) across epochs; use as proxy for learning activity.
- Feasibility limits:
  - Direct semantic interpretability of GRU hidden state is limited; we rely on probes/correlations.
  - Attention weights are available but currently discarded in `GraphAttentionEmbedding.aggregate` (returns weights but they’re ignored). Exposing and logging them requires a small code change (see temporal attention, soakn/tgn/model/temporal_attention.py:44).

4) Can we detect drift of customer points in embedding space?
- Statement: A drift detector over embeddings should flag changes around ground-truth τ.
- Data:
  - Embedding trajectories and cluster labels.
- Design:
  - For each customer, compute distance to start-centroid over time; apply change-point detection (CUSUM-like threshold on derivative or absolute change). Measure detection delay vs τ (absolute timestamp or normalized index) and false positives.
  - Aggregate per cluster: mean detection delay, recall/precision of drift events.
- Outputs:
  - Per-customer drift indices; per-cluster summary metrics.

5) Does the link-recurrent memory updater learn the synthetic decay functions?
- Statement: The learned scoring over time for a fixed customer–product pair should follow the synthetic decay’s shape (exponential, logistic, piecewise-exponential with α, τ), at least approximately.
- Data:
  - Predicted logits/probabilities for target pairs across evenly spaced timepoints (or event times) with memory frozen.
- Design:
  - For each cluster: choose a small set of canonical products (high mass in end distribution). For customers in that cluster, compute pos logits vs normalized time using `compute_edge_logits_many_negs_no_update` (soakn/tgn/model/tgn.py:106) and aggregate across customers.
  - Fit parametric curves to mean logits (or probabilities) vs normalized time; report R² and parameter error vs known α, τ when available.
  - Ablations: turn off memory (`--use_memory False`) or swap updater type (`--memory_updater rnn`), measure shape degradation.
- Caveats:
  - The GRU is not constrained to learn an analytic decay; evidence will be approximate (fit quality and monotonic trends), not a proof of identifiability.

6) Is link prediction accuracy time-sensitive as expected?
- Statement: Predictive performance should vary across time; time encoder and memory should matter.
- Data:
  - Standard val/test splits; additional evaluation over temporal bins.
- Design:
  - Slice val/test edges into time quantiles; compute AP/AUC per bin using `eval_edge_prediction` (soakn/tgn/evaluation/evaluation.py:6) on filtered `Data` objects.
  - Ablations: shuffle timestamps, disable time encoder (embedding_module=identity and/or `--use_memory False`), compare per-bin curves.

What To Extract and When

During Training (within epoch)
- Per-batch sampled embeddings:
  - For a fixed subset of customers per cluster, record embeddings at current batch timestamps with `_no_memory_side_effects()` to avoid modifying training state.
  - Store tuples: customer_node_id, cluster, timestamp, event_idx, embedding (np.array).
- Per-batch link logits (optional):
  - Save sampled positive logits for canonical products to build time–logit curves.
- Gradients (already partially present):
  - Grad-norms for memory updater, message function, aggregator are printed in train loop (soakn/tgn/train_self_supervised.py:294+). Capture into a CSV for later correlation.

End Of Epoch
- Memory snapshot:
  - Use `tgn.memory.backup_memory()` to capture memory, last_update, and messages; write memory as .npy per epoch.
- Embedding snapshot:
  - For all customers or a stratified sample, compute embeddings at standard timepoints (start/mid/end) with no side effects; write to parquet.
- Cluster centroids:
  - Compute per-cluster centroids and intra/extra-cluster distances; store metrics.

End Of Training
- Full evaluation across time bins:
  - Use filtered `Data` to compute AP/AUC over quantile bins; store per-bin metrics.
- Trajectories and drift:
  - Consolidate per-customer trajectories; run drift detection; store detections and delays vs τ.
- Decay-shape fit:
  - For each cluster/product, aggregate logit curves vs normalized time, fit parametric decays, store fit params and R².

Hook Points and Minimal Code Changes
- Freeze side effects during scoring:
  - Use `TGN._no_memory_side_effects()` (soakn/tgn/model/tgn.py:91) to compute embeddings/logits without mutating memory/messages.
- Get embeddings at times:
  - Call `TGN.compute_temporal_embeddings(sources, destinations, negatives, timestamps, edge_idxs, n_neighbors)` and take the first return (source embeddings). For pure customer embeddings, pass dummy `destinations`/`negatives` copied from `sources`.
- Attention weights (optional, requires change):
  - `TemporalAttentionLayer.forward` returns attention weights (soakn/tgn/model/temporal_attention.py:44). `GraphAttentionEmbedding.aggregate` currently discards them. To analyze attention:
    - Extend `aggregate` to return weights alongside embeddings and thread them up to a logging hook in `compute_embedding`.
- Centralized logging:
  - Add lightweight writers (CSV/NumPy) to the training loop for: per-epoch memory snapshot, sampled embeddings, gradient norms, per-bin eval metrics.

Ground-Truth and Metadata Alignment
- Node mapping: use `synthetic_node2id.pkl` to map customer/product to TGN node ids.
- Cluster labels: preserve the `cluster` column during preprocessing; join to customer node ids for supervised cluster metrics.
- Product categories: from products.csv, join on product_id to aggregate link scores by category.
- Decay parameters: from clusters.csv, join α, τ per cluster to analyze shape fits.

Outputs To Persist
- training_run_id/results.pkl: as already saved by train loop with AP, AUC, losses.
- embeddings/epoch_{k}.parquet: sampled embeddings with node_id, cluster, timestamp, event_idx.
- memory/epoch_{k}.npy: memory matrix; memory/epoch_{k}_last_update.npy.
- logits/cluster_{c}_product_{p}.csv: time vs mean logit for decay shape fits.
- metrics/cluster_quality.csv: silhouette, ARI/NMI per epoch and timepoint.
- metrics/drift_detection.csv: detection delays and rates per cluster.
- metrics/time_binned_eval.csv: AP/AUC per bin and ablation variants.

Feasibility Notes
- Exact recovery of analytic decay functions from a GRU memory is not expected. We aim for approximate shape consistency and parameter fits as evidence, not proof.
- Attention-level interpretability is possible but requires exposing/logging attention weights from the embedding module.
- All logging described above can be added with minimal overhead and without changing the learning dynamics if we use `_no_memory_side_effects()` for probes.

Next Steps
- Implement logging hooks in train_self_supervised.py at the batch and epoch boundaries as outlined.
- Add a helper in TGN to fetch node embeddings at arbitrary timestamps without side effects (thin wrapper over `compute_temporal_embeddings`).
- Add evaluation scripts/notebooks to consume the persisted outputs and compute the metrics described above.

