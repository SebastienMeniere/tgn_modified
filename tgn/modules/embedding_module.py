import math
import numpy as np
import torch
from torch import nn

from model.temporal_attention import TemporalAttentionLayer


class EmbeddingModule(nn.Module):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout, *, use_memory=True, memory_projector=None):
    super().__init__()
    self.node_features = node_features
    self.edge_features = edge_features
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device
    self.use_memory = use_memory
    
    if self.use_memory:
      self.memory_projector = memory_projector if memory_projector is not None else nn.Identity()
    else:
      self.memory_projector = None

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    raise NotImplementedError

  @staticmethod
  def _tensor_from_memory(memory_obj_or_tensor):
    if memory_obj_or_tensor is None:
      return None
    if isinstance(memory_obj_or_tensor, torch.Tensor):
      return memory_obj_or_tensor
    if hasattr(memory_obj_or_tensor, "memory"):
      return memory_obj_or_tensor.memory
    raise TypeError(f"Unsupported memory container: {type(memory_obj_or_tensor)}")


class IdentityEmbedding(EmbeddingModule):
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    if memory is None:
      raise ValueError("IdentityEmbedding requires memory to be enabled.")
    mem_tensor = self._tensor_from_memory(memory)
    idx = torch.as_tensor(source_nodes, device=mem_tensor.device, dtype=torch.long)
    emb = mem_tensor[idx, :]
    if self.memory_projector is not None:
      emb = self.memory_projector(emb)
    return emb


class TimeEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1, memory_projector=None):
    super().__init__(node_features, edge_features, memory,
                     neighbor_finder, time_encoder, n_layers,
                     n_node_features, n_edge_features, n_time_features,
                     embedding_dimension, device, dropout,
                     use_memory=use_memory, memory_projector=memory_projector)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.embedding_dimension)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    if memory is None:
      raise ValueError("TimeEmbedding requires memory to be enabled.")
    if time_diffs is None:
      raise ValueError("TimeEmbedding expects time_diffs tensor.")

    mem_tensor = self._tensor_from_memory(memory)
    idx = torch.as_tensor(source_nodes, device=mem_tensor.device, dtype=torch.long)
    mem = mem_tensor[idx, :]
    if self.memory_projector is not None:
      mem = self.memory_projector(mem)

    time_factor = self.embedding_layer(time_diffs.unsqueeze(1))
    source_embeddings = mem * (1 + time_factor)
    return source_embeddings


class GraphEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, memory_projector=None):
    super().__init__(node_features, edge_features, memory,
                     neighbor_finder, time_encoder, n_layers,
                     n_node_features, n_edge_features, n_time_features,
                     embedding_dimension, device, dropout,
                     use_memory=use_memory, memory_projector=memory_projector)

    self.use_memory = use_memory
    self.device = device
    
    # Project raw features to embedding dimension
    self.node_proj = nn.Linear(node_features.shape[1], embedding_dimension, bias=False)
    self.edge_proj = nn.Linear(n_edge_features, embedding_dimension, bias=False)
    self.time_proj = nn.Linear(n_time_features, embedding_dimension, bias=False)
    
    # Verify dimension compatibility
    print(f"[GraphEmbedding] Node features: {node_features.shape[1]} -> {embedding_dimension}")
    print(f"[GraphEmbedding] Edge features: {n_edge_features} -> {embedding_dimension}")
    print(f"[GraphEmbedding] Time features: {n_time_features} -> {embedding_dimension}")

  def _project_memory(self, memory, index_tensor):
    """Project memory to embedding dimension with proper error handling."""
    if not self.use_memory:
      return None
    mem_tensor = self._tensor_from_memory(memory)
    if mem_tensor is None:
      return None
    mem = mem_tensor[index_tensor]
    if self.memory_projector is not None:
      mem = self.memory_projector(mem)
    return mem

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    """
    Compute embeddings with fixed projection logic.
    
    Key fixes:
    1. Remove redundant h_src_conv computation
    2. Use h_src (with memory) directly in aggregation
    3. Ensure all projections happen before aggregation
    4. Add dimension assertions for debugging
    """
    assert n_layers >= 0

    # Convert inputs to tensors
    src_t = torch.from_numpy(source_nodes).long().to(self.device)
    ts_t = torch.from_numpy(timestamps).float().to(self.device).unsqueeze(1)

    # Step 1: Project raw node features to embedding dimension
    h_src = self.node_proj(self.node_features[src_t])
    
    # Step 2: Add projected memory if available
    if self.use_memory:
      mem = self._project_memory(memory, src_t)
      if mem is not None:
        h_src = h_src + mem
    
    # Dimension check
    assert h_src.shape[-1] == self.embedding_dimension, \
      f"h_src dimension mismatch: {h_src.shape[-1]} != {self.embedding_dimension}"

    # Base case: no graph convolution layers
    if n_layers == 0:
      return h_src

    # Step 3: Get temporal neighbors
    neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
      source_nodes,
      timestamps,
      n_neighbors=n_neighbors)

    neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
    edge_idxs_torch = torch.from_numpy(edge_idxs).long().to(self.device)

    # Step 4: Compute edge time deltas
    edge_deltas = timestamps[:, np.newaxis] - edge_times
    edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

    # Step 5: Recursively compute neighbor embeddings
    neigh_flat = neighbors.flatten()
    neighbor_embeddings = self.compute_embedding(
      memory,
      neigh_flat,
      np.repeat(timestamps, n_neighbors),
      n_layers=n_layers - 1,
      n_neighbors=n_neighbors
    )

    effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)

    # Step 6: Project edge features and time embeddings
    edge_time_embeddings = self.time_proj(self.time_encoder(edge_deltas_torch))
    edge_features = self.edge_proj(self.edge_features[edge_idxs_torch, :])
    
    # Step 7: Compute source node time embedding (Δt = 0 for query node)
    src_time_emb = self.time_proj(self.time_encoder(torch.zeros_like(ts_t)))

    # Dimension checks
    assert edge_time_embeddings.shape[-1] == self.embedding_dimension, \
      f"Edge time embedding dimension mismatch: {edge_time_embeddings.shape[-1]} != {self.embedding_dimension}"
    assert edge_features.shape[-1] == self.embedding_dimension, \
      f"Edge feature dimension mismatch: {edge_features.shape[-1]} != {self.embedding_dimension}"
    assert neighbor_embeddings.shape[-1] == self.embedding_dimension, \
      f"Neighbor embedding dimension mismatch: {neighbor_embeddings.shape[-1]} != {self.embedding_dimension}"
    assert src_time_emb.shape[-1] == self.embedding_dimension, \
      f"Source time embedding dimension mismatch: {src_time_emb.shape[-1]} != {self.embedding_dimension}"

    # Step 8: Create mask for padding
    mask = neighbors_torch == 0

    # Step 9: Aggregate using the appropriate strategy
    # CRITICAL FIX: Pass h_src directly (already includes memory)
    source_embedding = self.aggregate(
      n_layers, 
      h_src,  # Use h_src which already has memory added
      src_time_emb,
      neighbor_embeddings,
      edge_time_embeddings,
      edge_features,
      mask
    )

    return source_embedding

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings, edge_time_embeddings, edge_features, mask):
    raise NotImplementedError


class GraphSumEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, memory_projector=None):
    super().__init__(node_features=node_features,
                     edge_features=edge_features,
                     memory=memory,
                     neighbor_finder=neighbor_finder,
                     time_encoder=time_encoder, n_layers=n_layers,
                     n_node_features=n_node_features,
                     n_edge_features=n_edge_features,
                     n_time_features=n_time_features,
                     embedding_dimension=embedding_dimension,
                     device=device,
                     n_heads=n_heads, dropout=dropout,
                     use_memory=use_memory,
                     memory_projector=memory_projector)

    D = embedding_dimension
    
    # Layer 1: Project concatenated neighbor features [3*D] -> [D]
    self.linear_1 = nn.ModuleList([
      nn.Linear(3 * D, D) for _ in range(n_layers)
    ])
    
    # Layer 2: Project concatenated source and neighbor features [3*D] -> [D]
    self.linear_2 = nn.ModuleList([
      nn.Linear(3 * D, D) for _ in range(n_layers)
    ])
    
    print(f"[GraphSumEmbedding] Created {n_layers} aggregation layers with dimension {D}")

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings, edge_time_embeddings, edge_features, mask):
    """
    Aggregate neighbor information using summation.
    
    Fixed to ensure dimension consistency:
    - All inputs should be in embedding_dimension
    - Concatenations are done correctly
    - Layer normalization is applied at the end
    """
    # Verify dimensions
    D = self.embedding_dimension
    assert source_node_features.shape[-1] == D, \
      f"Source node features dim: {source_node_features.shape[-1]} != {D}"
    assert source_nodes_time_embedding.shape[-1] == D, \
      f"Source time embedding dim: {source_nodes_time_embedding.shape[-1]} != {D}"
    assert neighbor_embeddings.shape[-1] == D, \
      f"Neighbor embeddings dim: {neighbor_embeddings.shape[-1]} != {D}"
    assert edge_time_embeddings.shape[-1] == D, \
      f"Edge time embeddings dim: {edge_time_embeddings.shape[-1]} != {D}"
    assert edge_features.shape[-1] == D, \
      f"Edge features dim: {edge_features.shape[-1]} != {D}"
    
    # Concatenate neighbor features: [B, N, D] + [B, N, D] + [B, N, D] -> [B, N, 3*D]
    neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features], dim=2)
    
    # Project neighbor features: [B, N, 3*D] -> [B, N, D]
    neighbor_embeddings_proj = self.linear_1[n_layer - 1](neighbors_features)
    
    # Apply mask before averaging (set padded neighbors to zero)
    neighbor_embeddings_proj = neighbor_embeddings_proj.masked_fill(mask.unsqueeze(-1), 0.0)
    
    # Average over neighbors: [B, N, D] -> [B, D]
    # Use sum and divide by number of non-masked neighbors for proper averaging
    neighbor_count = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
    neighbors_sum = neighbor_embeddings_proj.sum(dim=1)
    neighbors_mean = neighbors_sum / neighbor_count
    neighbors_mean = torch.nn.functional.relu(neighbors_mean)
    
    # Concatenate source features: [B, D] + [B, D] -> [B, 2*D]
    source_features = torch.cat([
      source_node_features,
      source_nodes_time_embedding.squeeze(1)
    ], dim=1)
    
    # Concatenate all: [B, D] + [B, 2*D] -> [B, 3*D]
    combined = torch.cat([neighbors_mean, source_features], dim=1)
    
    # Final projection: [B, 3*D] -> [B, D]
    source_embedding = self.linear_2[n_layer - 1](combined)
    
    # Apply layer normalization
    source_embedding = torch.nn.functional.layer_norm(
      source_embedding, source_embedding.shape[-1:]
    )
    
    return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, memory_projector=None):
    super().__init__(node_features=node_features,
                     edge_features=edge_features,
                     memory=memory,
                     neighbor_finder=neighbor_finder,
                     time_encoder=time_encoder, n_layers=n_layers,
                     n_node_features=n_node_features,
                     n_edge_features=n_edge_features,
                     n_time_features=n_time_features,
                     embedding_dimension=embedding_dimension,
                     device=device,
                     n_heads=n_heads, dropout=dropout,
                     use_memory=use_memory,
                     memory_projector=memory_projector)

    self.attention_models = nn.ModuleList([TemporalAttentionLayer(
      n_node_features=embedding_dimension,
      n_neighbors_features=embedding_dimension,
      n_edge_features=embedding_dimension,
      time_dim=embedding_dimension,
      n_head=n_heads,
      dropout=dropout,
      output_dimension=embedding_dimension)
      for _ in range(n_layers)])
    
    print(f"[GraphAttentionEmbedding] Created {n_layers} attention layers with {n_heads} heads")

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings, edge_time_embeddings, edge_features, mask):
    """
    Aggregate neighbor information using temporal attention.
    
    All inputs are already projected to embedding_dimension.
    """
    # Verify dimensions
    D = self.embedding_dimension
    assert source_node_features.shape[-1] == D, \
      f"Source node features dim: {source_node_features.shape[-1]} != {D}"
    assert source_nodes_time_embedding.shape[-1] == D, \
      f"Source time embedding dim: {source_nodes_time_embedding.shape[-1]} != {D}"
    assert neighbor_embeddings.shape[-1] == D, \
      f"Neighbor embeddings dim: {neighbor_embeddings.shape[-1]} != {D}"
    assert edge_time_embeddings.shape[-1] == D, \
      f"Edge time embeddings dim: {edge_time_embeddings.shape[-1]} != {D}"
    assert edge_features.shape[-1] == D, \
      f"Edge features dim: {edge_features.shape[-1]} != {D}"
    
    attention_model = self.attention_models[n_layer - 1]
    
    source_embedding, _ = attention_model(
      source_node_features,
      source_nodes_time_embedding,
      neighbor_embeddings,
      edge_time_embeddings,
      edge_features,
      mask
    )
    
    # Apply layer normalization
    source_embedding = torch.nn.functional.layer_norm(
      source_embedding, source_embedding.shape[-1:]
    )
    
    return source_embedding


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True, memory_projector=None):
  """
  Factory function to create embedding modules.
  
  Added validation to ensure dimension consistency.
  """
  
  # Validate inputs
  print(f"\n[get_embedding_module] Creating {module_type} module")
  print(f"  Node features: {n_node_features}")
  print(f"  Edge features: {n_edge_features}")
  print(f"  Time features: {n_time_features}")
  print(f"  Embedding dimension: {embedding_dimension}")
  print(f"  Use memory: {use_memory}")
  
  if use_memory and memory_projector is not None:
    print(f"  Memory projector: {type(memory_projector).__name__}")
  
  if module_type == "graph_attention":
    return GraphAttentionEmbedding(node_features=node_features,
                                   edge_features=edge_features,
                                   memory=memory,
                                   neighbor_finder=neighbor_finder,
                                   time_encoder=time_encoder,
                                   n_layers=n_layers,
                                   n_node_features=n_node_features,
                                   n_edge_features=n_edge_features,
                                   n_time_features=n_time_features,
                                   embedding_dimension=embedding_dimension,
                                   device=device,
                                   n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                                   memory_projector=memory_projector)
  elif module_type == "graph_sum":
    return GraphSumEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                             memory_projector=memory_projector)
  elif module_type == "identity":
    return IdentityEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout,
                             use_memory=use_memory,
                             memory_projector=memory_projector)
  elif module_type == "time":
    return TimeEmbedding(node_features=node_features,
                         edge_features=edge_features,
                         memory=memory,
                         neighbor_finder=neighbor_finder,
                         time_encoder=time_encoder,
                         n_layers=n_layers,
                         n_node_features=n_node_features,
                         n_edge_features=n_edge_features,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         dropout=dropout,
                         use_memory=use_memory,
                         n_neighbors=n_neighbors,
                         memory_projector=memory_projector)
  else:
    raise ValueError(f"Embedding Module {module_type} not supported")