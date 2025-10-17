import bisect
from collections import defaultdict
import numpy as np
import torch


def _fast_unique(values):
  """
  More scalable unique helper that avoids the O(n log n) numpy sort on large arrays
  by delegating to torch.unique when we have millions of entries.
  """
  arr = np.asarray(values)

  if arr.ndim == 0:
    return arr.reshape(1)

  if arr.size > 2_000_000 and arr.dtype.kind in {"i", "u", "f"}:
    tensor = torch.as_tensor(arr)
    return torch.unique(tensor, sorted=False).cpu().numpy()

  return np.unique(arr)


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class MLP(torch.nn.Module):
  def __init__(self, dim_in, hd_1, hd_2, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim_in, hd_1)
    self.fc_2 = torch.nn.Linear(hd_1, hd_2)
    self.fc_3 = torch.nn.Linear(hd_2, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = _fast_unique(src_list)
    self.dst_list = _fast_unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


class HistoryEdgeSampler:
  """
  Negative sampler that avoids edges already observed before the current timestamp for each source.
  """

  def __init__(self, src, dst, ts, seed=None):
    self.seed = seed
    self.rng = np.random.RandomState(seed)
    self.all_dst = _fast_unique(dst)

    self.hist = defaultdict(list)
    for s, d, t in zip(src, dst, ts):
      self.hist[s].append((t, d))
    for s in self.hist:
      self.hist[s].sort(key=lambda x: x[0])

  def reset_random_state(self):
    self.rng = np.random.RandomState(self.seed)

  def _seen_before(self, s, d, t):
    pair_list = self.hist.get(s)
    if not pair_list:
      return False
    idx = bisect.bisect_left(pair_list, (t, -1))
    for _, dest in pair_list[:idx]:
      if dest == d:
        return True
    return False

  def sample_batch(self, src_batch, ts_batch, num_neg=1):
    src_batch = np.asarray(src_batch)
    ts_batch = np.asarray(ts_batch)
    n = len(src_batch)
    if num_neg <= 1:
      negatives = np.empty(n, dtype=self.all_dst.dtype)
    else:
      negatives = np.empty((n, num_neg), dtype=self.all_dst.dtype)

    for i, (s, t) in enumerate(zip(src_batch, ts_batch)):
      for k in range(num_neg):
        tries = 0
        while True:
          d = self.rng.choice(self.all_dst)
          if not self._seen_before(s, d, t) or tries >= len(self.all_dst):
            if num_neg <= 1:
              negatives[i] = d
            else:
              negatives[i, k] = d
            break
          tries += 1

    return src_batch, negatives

class FilteredRandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None, existing_edges=None):
        """
        Enhanced edge sampler that filters out existing edges to avoid false negatives
        
        Args:
            src_list: list of source nodes
            dst_list: list of destination nodes  
            seed: random seed for reproducibility
            existing_edges: optional set of (src, dst) tuples representing existing edges
                          if None, will be computed from src_list and dst_list
        """
        self.seed = seed
        self.src_list = _fast_unique(src_list)
        self.dst_list = _fast_unique(dst_list)
        
        if seed is not None:
            self.random_state = np.random.RandomState(self.seed)
        
        # Create set of existing edges for fast lookup
        if existing_edges is None:
            self.existing_edges = set(zip(src_list, dst_list))
        else:
            self.existing_edges = existing_edges
            
        print(f"FilteredRandEdgeSampler initialized with {len(self.existing_edges)} existing edges")
        print(f"Unique sources: {len(self.src_list)}, Unique destinations: {len(self.dst_list)}")
        
        # Pre-compute possible negatives for efficiency (optional optimization)
        self._precompute_negatives = len(self.src_list) * len(self.dst_list) < 1000000  # Only for smaller graphs
        if self._precompute_negatives:
          print("computing negatives")
          self._compute_all_possible_negatives()
    
    def _compute_all_possible_negatives(self):
        """Pre-compute all possible negative edges for very fast sampling"""
        all_possible = set()
        for src in self.src_list:
            for dst in self.dst_list:
                if (src, dst) not in self.existing_edges:
                    all_possible.add((src, dst))
        
        self.possible_negatives = list(all_possible)
        print(f"Pre-computed {len(self.possible_negatives)} possible negative edges")
    
    def sample(self, size):
        """
        Sample negative edges that don't exist in the training data
        
        Args:
            size: number of negative edges to sample
            
        Returns:
            tuple: (source_nodes, destination_nodes)
        """
        if self._precompute_negatives and hasattr(self, 'possible_negatives'):
            return self._sample_from_precomputed(size)
        else:
            return self._sample_with_rejection(size)
    
    def _sample_from_precomputed(self, size):
        """Fast sampling from pre-computed negatives"""
        if len(self.possible_negatives) < size:
            print(f"Warning: Requested {size} negatives but only {len(self.possible_negatives)} possible")
            size = len(self.possible_negatives)
        
        if self.seed is None:
            indices = np.random.choice(len(self.possible_negatives), size=size, replace=False)
        else:
            indices = self.random_state.choice(len(self.possible_negatives), size=size, replace=False)
        
        sampled_edges = [self.possible_negatives[i] for i in indices]
        sources, destinations = zip(*sampled_edges)
        
        return np.array(sources), np.array(destinations)
    
    def _sample_with_rejection(self, size):
        """Sample using rejection sampling - slower but memory efficient"""
        sources = []
        destinations = []
        
        max_attempts = size * 100  # Prevent infinite loops
        attempts = 0
        
        while len(sources) < size and attempts < max_attempts:
            if self.seed is None:
                src_idx = np.random.randint(0, len(self.src_list))
                dst_idx = np.random.randint(0, len(self.dst_list))
            else:
                src_idx = self.random_state.randint(0, len(self.src_list))
                dst_idx = self.random_state.randint(0, len(self.dst_list))
            
            src = self.src_list[src_idx]
            dst = self.dst_list[dst_idx]
            
            if (src, dst) not in self.existing_edges:
                sources.append(src)
                destinations.append(dst)
            
            attempts += 1
        
        if len(sources) < size:
            print(f"Warning: Could only sample {len(sources)} negatives out of {size} requested")
        
        return np.array(sources), np.array(destinations)
    
    def reset_random_state(self):
        """Reset the random state - maintains compatibility with original interface"""
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

# Alternative: Advanced negative sampler with additional strategies
class AdvancedNegativeSampler(object):
    def __init__(self, src_list, dst_list, timestamps=None, seed=None, strategy='filtered_random'):
        """
        Advanced negative sampler with multiple strategies
        
        Args:
            src_list: list of source nodes
            dst_list: list of destination nodes
            timestamps: optional timestamps for temporal-aware sampling
            seed: random seed
            strategy: 'filtered_random', 'popularity_biased', 'temporal_aware'
        """
        self.seed = seed
        self.src_list = _fast_unique(src_list)
        self.dst_list = _fast_unique(dst_list)
        self.strategy = strategy
        self.existing_edges = set(zip(src_list, dst_list))
        
        if seed is not None:
            self.random_state = np.random.RandomState(self.seed)
        
        # Compute popularity for biased sampling
        if strategy in ['popularity_biased', 'temporal_aware']:
            self._compute_popularity(dst_list)
        
        # Compute temporal patterns
        if strategy == 'temporal_aware' and timestamps is not None:
            self._compute_temporal_patterns(src_list, dst_list, timestamps)
    
    def _compute_popularity(self, dst_list):
        """Compute destination node popularity for biased sampling"""
        dst_counts = defaultdict(int)
        for dst in dst_list:
            dst_counts[dst] += 1
        
        # Convert to probabilities
        total = sum(dst_counts.values())
        self.dst_probs = {}
        for dst in self.dst_list:
            self.dst_probs[dst] = dst_counts[dst] / total
    
    def _compute_temporal_patterns(self, src_list, dst_list, timestamps):
        """Compute temporal interaction patterns"""
        # Group interactions by time windows
        self.temporal_groups = defaultdict(set)
        time_window = 3600  # 1 hour windows, adjust as needed
        
        for src, dst, ts in zip(src_list, dst_list, timestamps):
            window = ts // time_window
            self.temporal_groups[window].add((src, dst))
    
    def sample(self, size):
        """Sample negative edges based on the chosen strategy"""
        if self.strategy == 'filtered_random':
            return self._sample_filtered_random(size)
        elif self.strategy == 'popularity_biased':
            return self._sample_popularity_biased(size)
        elif self.strategy == 'temporal_aware':
            return self._sample_temporal_aware(size)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _sample_filtered_random(self, size):
        """Standard filtered random sampling"""
        sources, destinations = [], []
        max_attempts = size * 100
        attempts = 0
        
        while len(sources) < size and attempts < max_attempts:
            if self.seed is None:
                src = np.random.choice(self.src_list)
                dst = np.random.choice(self.dst_list)
            else:
                src = self.random_state.choice(self.src_list)
                dst = self.random_state.choice(self.dst_list)
            
            if (src, dst) not in self.existing_edges:
                sources.append(src)
                destinations.append(dst)
            attempts += 1
        
        return np.array(sources), np.array(destinations)
    
    def _sample_popularity_biased(self, size):
        """Sample negatives biased towards popular items (harder negatives)"""
        sources, destinations = [], []
        dst_list = list(self.dst_probs.keys())
        dst_weights = list(self.dst_probs.values())
        
        max_attempts = size * 100
        attempts = 0
        
        while len(sources) < size and attempts < max_attempts:
            if self.seed is None:
                src = np.random.choice(self.src_list)
                dst = np.random.choice(dst_list, p=dst_weights)
            else:
                src = self.random_state.choice(self.src_list)
                dst = self.random_state.choice(dst_list, p=dst_weights)
            
            if (src, dst) not in self.existing_edges:
                sources.append(src)
                destinations.append(dst)
            attempts += 1
        
        return np.array(sources), np.array(destinations)
    
    def _sample_temporal_aware(self, size):
        """Sample negatives that are temporally relevant but don't exist"""
        # Simplified version - can be enhanced based on specific temporal requirements
        return self._sample_popularity_biased(size)
    
    def reset_random_state(self):
        """Reset the random state"""
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times
