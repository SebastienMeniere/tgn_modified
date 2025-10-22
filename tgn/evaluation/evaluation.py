import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Optional
try:
  from tqdm.auto import tqdm
except Exception:
  tqdm = None


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200,
                         *, progress: bool = False, desc: Optional[str] = None,
                         check_false_negatives: bool = True):
    """
    Evaluate edge prediction with proper negative sampling validation.
    
    Args:
        model: TGN model
        negative_edge_sampler: Sampler for negative edges (should be HistoryEdgeSampler)
        data: Evaluation data
        n_neighbors: Number of neighbors for embedding computation
        batch_size: Batch size for evaluation
        progress: Show progress bar
        desc: Description for progress bar
        check_false_negatives: If True, validate that negatives don't exist in eval set
    
    Returns:
        mean_ap: Mean average precision
        mean_auc: Mean area under ROC curve
    """
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None, "Negative sampler must have a seed for reproducible evaluation"
    negative_edge_sampler.reset_random_state()
    
    # CRITICAL: Check if sampler can avoid false negatives
    if check_false_negatives and not hasattr(negative_edge_sampler, 'sample_batch'):
        print("WARNING: Using a sampler without temporal awareness (sample_batch method).")
        print("This may include false negatives (real edges labeled as negative)!")
        print("Recommend using HistoryEdgeSampler for proper evaluation.")
    
    # Create set of edges in evaluation data for validation
    eval_edges = set(zip(data.sources, data.destinations))
    false_negative_count = 0
    total_negatives = 0
    
    val_ap, val_auc = [], []
    
    with torch.no_grad():
        model = model.eval()
        
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        
        iterator = range(num_test_batch)
        if progress and tqdm is not None:
            iterator = tqdm(iterator, total=num_test_batch, desc=desc or "Eval", leave=False)
        
        for k in iterator:
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx: e_idx]
            
            size = len(sources_batch)
            
            # Sample negative edges
            if hasattr(negative_edge_sampler, "sample_batch"):
                # HistoryEdgeSampler - temporally aware, avoids seen edges
                _, negative_samples = negative_edge_sampler.sample_batch(
                    sources_batch, timestamps_batch, num_neg=1
                )
                if negative_samples.ndim > 1:
                    negative_samples = negative_samples[:, 0]
            else:
                # RandEdgeSampler - may include false negatives
                _, negative_samples = negative_edge_sampler.sample(size)
            
            # VALIDATION: Check for false negatives (optional but recommended)
            if check_false_negatives:
                for src, neg in zip(sources_batch, negative_samples):
                    total_negatives += 1
                    if (src, neg) in eval_edges:
                        false_negative_count += 1
            
            # Compute edge probabilities
            pos_prob, neg_prob = model.compute_edge_probabilities(
                sources_batch, 
                destinations_batch,
                negative_samples, 
                timestamps_batch,
                edge_idxs_batch, 
                n_neighbors
            )
            
            # Compute metrics
            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
            
            if not progress:
                print(f"Batch {k} of {num_test_batch}: AP={val_ap[-1]:.4f}, AUC={val_auc[-1]:.4f}")
    
    mean_ap = np.mean(val_ap)
    mean_auc = np.mean(val_auc)
    
    # Report false negative statistics
    if check_false_negatives and total_negatives > 0:
        false_neg_rate = 100 * false_negative_count / total_negatives
        if false_negative_count > 0:
            print(f"\n⚠️  WARNING: Found {false_negative_count}/{total_negatives} false negatives ({false_neg_rate:.2f}%)")
            print("These are real edges that were sampled as negatives!")
            print("Metrics may be inflated. Consider using HistoryEdgeSampler.")
        else:
            print(f"✓ No false negatives detected ({total_negatives} negatives checked)")
    
    if not progress:
        print(f"\nFinal Results: Mean AP={mean_ap:.4f}, Mean AUC={mean_auc:.4f}")
    
    return mean_ap, mean_auc


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc
