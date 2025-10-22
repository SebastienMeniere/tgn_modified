from torch import nn
import torch


class MemoryUpdater(nn.Module):
  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    pass


class SequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(SequenceMemoryUpdater, self).__init__()
    self.memory = memory
    self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.memory_dimension = memory_dimension
    self.device = device
    
    print(f"[SequenceMemoryUpdater] Message dim: {message_dimension}, Memory dim: {memory_dimension}")

  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), \
      "Trying to update memory to time in the past"

    # Get current memory
    memory = self.memory.get_memory(unique_node_ids)
    
    # Validate dimensions
    assert unique_messages.shape[-1] == self.message_dimension, \
      f"Message dimension mismatch: {unique_messages.shape[-1]} != {self.message_dimension}"
    assert memory.shape[-1] == self.memory_dimension, \
      f"Memory dimension mismatch: {memory.shape[-1]} != {self.memory_dimension}"
    
    # Update timestamp
    self.memory.last_update[unique_node_ids] = timestamps

    # Apply memory update function (GRU/RNN)
    updated_memory = self.memory_updater(unique_messages, memory)
    
    # CRITICAL FIX: Apply layer normalization
    updated_memory = self.layer_norm(updated_memory)

    # Store updated memory
    self.memory.set_memory(unique_node_ids, updated_memory)

  def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
    """
    Get updated memory without modifying the actual memory state.
    Used for validation/inference where we don't want side effects.
    """
    if len(unique_node_ids) <= 0:
      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), \
      "Trying to update memory to time in the past"

    # Clone current state
    updated_memory = self.memory.memory.data.clone()
    
    # Get current memory for the nodes we're updating
    current_memory = updated_memory[unique_node_ids]
    
    # Validate dimensions
    assert unique_messages.shape[-1] == self.message_dimension, \
      f"Message dimension mismatch: {unique_messages.shape[-1]} != {self.message_dimension}"
    assert current_memory.shape[-1] == self.memory_dimension, \
      f"Memory dimension mismatch: {current_memory.shape[-1]} != {self.memory_dimension}"
    
    # Apply update
    new_memory = self.memory_updater(unique_messages, current_memory)
    
    # CRITICAL FIX: Apply layer normalization
    new_memory = self.layer_norm(new_memory)
    
    # Update the cloned memory
    updated_memory[unique_node_ids] = new_memory

    # Update timestamps
    updated_last_update = self.memory.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps
    
    return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)
    print(f"[GRUMemoryUpdater] Initialized with input_size={message_dimension}, hidden_size={memory_dimension}")


class RNNMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)
    print(f"[RNNMemoryUpdater] Initialized with input_size={message_dimension}, hidden_size={memory_dimension}")


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
  print(f"\n[get_memory_updater] Creating {module_type} updater")
  
  if module_type == "gru":
    return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
  elif module_type == "rnn":
    return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
  else:
    raise ValueError(f"Memory updater {module_type} not implemented")