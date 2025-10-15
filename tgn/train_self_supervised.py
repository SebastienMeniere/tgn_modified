import json
import math
import logging
import time
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import AdvancedNegativeSampler, EarlyStopMonitor, FilteredRandEdgeSampler, RandEdgeSampler, get_neighbor_finder
from utils.experiment_logging import TGNExperimentLogger, SnapshotConfig
from utils.data_processing import get_data, compute_time_statistics

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--data_dir', type=str, dest='data_dir', default='../data/processed',
                    help='Directory containing preprocessed dataset files (CSV/NPY).')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=1, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=1, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--patience', type=int, default=1, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.0, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=64, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=64, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=64, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=64, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--num_neg', type=int, default=1,
                    help='Number of negative samples per positive interaction')
parser.add_argument('--use_weighted_bce', action='store_true',
                    help='Use multi-negative weighted BCE that keeps memory frozen during scoring')
parser.add_argument('--neg_loss_weight', type=float, default=1.0,
                    help='Per-negative weight used by the weighted BCE branch (ignored otherwise)')
parser.add_argument('--log_dir', type=str, default='experiment_logs', help='Directory for experiment logs')
parser.add_argument('--snapshot', action='store_true', help='Enable per-epoch snapshots (embeddings/memory)')
parser.add_argument('--snapshot_nodes', type=int, default=256, help='Max number of customer nodes to snapshot')
parser.add_argument('--snapshot_quantiles', type=int, default=3, help='Number of quantile timepoints per node to snapshot')
parser.add_argument('--snapshot_quantile_mode', type=str, default='per_node', choices=['per_node','global'], help='Quantile mode for snapshot timepoints')
parser.add_argument('--snapshot_full_timeline', action='store_true', help='Use full neighbor finder and full-data timestamps for snapshotting')
parser.add_argument('--adv_sampler', action='store_true', help='Turning the advanced sampler on or off')
parser.add_argument('--negative_sampler', type=str, default='random',
                    choices=['random', 'advanced'],
                    help='Negative sampling backend: fast random (default) or advanced strategies.')
parser.add_argument('--adv_type', type=str, default='filtered_random',
                    choices=['filtered_random', 'popularity_biased', 'temporal_aware'],
                    help='Advanced sampler strategy (only used when --negative_sampler=advanced).')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

DATA_DIR = Path(args.data_dir).expanduser()
if not DATA_DIR.exists():
  raise SystemExit(f"Data directory '{DATA_DIR}' does not exist")

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = max(1, args.num_neg)
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
ADV_SAMP = args.adv_sampler
SAMPLER_TYPE = args.negative_sampler
ADV_TYPE = args.adv_type

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}.pth'
MODEL_METADATA_PATH = Path(MODEL_SAVE_PATH).with_suffix('.json')
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(
  DATA,
  different_new_nodes_between_val_and_test=args.different_new_nodes,
  randomize_features=args.randomize_features,
  data_dir=DATA_DIR,
)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# if ADV_SAMP:
#   sampler = RandEdgeSampler()
# else


# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs. In the inductive setting, negatives are sampled only amongst other new nodes.
def build_sampler(src, dst, seed=None):
  if SAMPLER_TYPE == 'advanced':
    return AdvancedNegativeSampler(src, dst, seed=seed, strategy=ADV_TYPE)
  return RandEdgeSampler(src, dst, seed=seed)


train_rand_sampler = build_sampler(train_data.sources, train_data.destinations)
val_rand_sampler = build_sampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = build_sampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
test_rand_sampler = build_sampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = build_sampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)
# # Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)
  node_mapping_path = Path(args.data_dir) / f"{DATA}_node2id.pkl"
  labels_path = Path(args.data_dir) / f"{DATA}_labels.csv"
  exp_logger = TGNExperimentLogger(SnapshotConfig(
    log_dir=Path(args.log_dir),
    run_prefix=args.prefix,
    snapshot_nodes=args.snapshot_nodes,
    node_mapping_path=node_mapping_path,
    labels_path=labels_path,
  ))

  # Initialize Model
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep,
            node_dimension=NODE_DIM,
            time_dimension=TIME_DIM)
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  best_checkpoint_path = None
  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    ### Training

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    logger.info('start {} epoch'.format(epoch))
    total_steps = math.ceil(num_batch / args.backprop_every)
    exp_logger.start_epoch_bar(epoch=epoch, total_steps=total_steps, desc=f"Train {epoch+1}/{NUM_EPOCH}")
    for step_idx, k in enumerate(range(0, num_batch, args.backprop_every)):
      optimizer.zero_grad()
      micro_losses = []
      commit_batches = []

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]

        size = len(sources_batch)
        neg_sample_size = size * NUM_NEG if NUM_NEG > 1 else size
        _, sampled_negatives = train_rand_sampler.sample(neg_sample_size)
        negatives_batch = sampled_negatives.reshape(size, NUM_NEG) if NUM_NEG > 1 else sampled_negatives

        tgn = tgn.train()

        if args.use_weighted_bce:
          # New path: gather logits with frozen memory, apply weighted BCE with K negatives per positive.
          # print("getting edges to compute probabilities")
          pos_logits, neg_logits = tgn.compute_edge_logits_many_negs_no_update(
              sources_batch,
              destinations_batch,
              negatives_batch,
              timestamps_batch,
              edge_idxs_batch,
              NUM_NEIGHBORS,
          )
          pos_logits = pos_logits.view(-1)
          neg_logits = neg_logits.view(size * NUM_NEG)
          logits = torch.cat([pos_logits, neg_logits], dim=0)
          targets = torch.cat([
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits)
          ], dim=0)
          weights = torch.cat([
            torch.full_like(pos_logits, 1.0),
            torch.full_like(neg_logits, args.neg_loss_weight)
          ], dim=0)
          micro_loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
        else:
          # Original computation retained as fallback (single negative per positive by default).
          # pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
          #                                                     timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
          pos_logits, neg_logits = tgn.compute_edge_logits_many_negs_no_update(
              sources_batch,
              destinations_batch,
              negatives_batch,
              timestamps_batch,
              edge_idxs_batch,
              NUM_NEIGHBORS,
          )
          pos_prob = torch.sigmoid(pos_logits)
          neg_prob = torch.sigmoid(neg_logits)
          w_pos = 1.0
          w_neg = 1.0   # or >1.0 if you want to emphasize negatives; you can also scale by K
          pos_loss = - (w_pos * torch.log(pos_prob.clamp_min(1e-12))).mean()
          neg_loss = - (w_neg * torch.log((1.0 - neg_prob).clamp_min(1e-12))).mean()
          micro_loss = pos_loss + neg_loss

        micro_losses.append(micro_loss)
        commit_batches.append((sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch))

      if not micro_losses:
        logger.debug("skipping empty micro-batch window")
        continue

      loss = torch.stack(micro_losses).mean()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      for name, param in tgn.named_parameters():
        if param.grad is None:

          continue
        if "memory_updater" in name:
          logger.info(f"{name} grad_norm={param.grad.norm().item():.4e}")
        if "message_function" in name:
          logger.info(f"{name} grad_norm={param.grad.norm().item():.4e}")
        if "message_aggregator" in name:
          logger.info(f"{name} grad_norm={param.grad.norm().item():.4e}")
      
      m_loss.append(loss.item())
      exp_logger.update_epoch_bar(loss_value=loss.item())

      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
      # the start of time
      if USE_MEMORY:
        tgn.memory.detach_memory()
        with torch.no_grad():
          for src_b, dst_b, ts_b, edge_idx_b in commit_batches:
            tgn.commit_positive_update(
                src_b, dst_b, ts_b,
                edge_idx_b, NUM_NEIGHBORS
            )

    exp_logger.close_epoch_bar()
    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    # Snapshots (after training phase, before eval)
    if args.snapshot:
      try:
        # memory snapshot always reflects current train phase
        exp_logger.snapshot_memory(tgn, epoch)

        # Optionally switch to full timeline for embedding snapshots
        if args.snapshot_full_timeline:
          tgn.set_neighbor_finder(full_ngh_finder)
          global_ts = full_data.timestamps if args.snapshot_quantile_mode == 'global' else None
        else:
          global_ts = train_data.timestamps if args.snapshot_quantile_mode == 'global' else None

        exp_logger.snapshot_embeddings(
          tgn,
          train_data,
          epoch,
          NUM_NEIGHBORS,
          max_nodes=args.snapshot_nodes,
          K=args.snapshot_quantiles,
          quantile_mode=args.snapshot_quantile_mode,
          global_ts=global_ts,
        )
      except Exception as e:
        logger.warning(f"Snapshotting failed at epoch {epoch}: {e}")

    ### Validation
    # Validation uses the full graph
    tgn.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
      train_memory_backup = tgn.memory.backup_memory()

    val_ap, val_auc = eval_edge_prediction(model=tgn,
                                                            negative_edge_sampler=val_rand_sampler,
                                                            data=val_data,
                                                            n_neighbors=NUM_NEIGHBORS,
                                                            progress=True, desc=f"Val {epoch+1}/{NUM_EPOCH}")
    if USE_MEMORY:
      val_memory_backup = tgn.memory.backup_memory()
      # Restore memory we had at the end of training to be used when validating on new nodes.
      # Also backup memory after validation so it can be used for testing (since test edges are
      # strictly later in time than validation edges)
      tgn.memory.restore_memory(train_memory_backup)

    # Validate on unseen nodes
    nn_val_ap, nn_val_auc = eval_edge_prediction(model=tgn,
                                                                        negative_edge_sampler=val_rand_sampler,
                                                                        data=new_node_val_data,
                                                                        n_neighbors=NUM_NEIGHBORS,
                                                                        progress=True, desc=f"NewNode Val {epoch+1}/{NUM_EPOCH}")

    if USE_MEMORY:
      # Restore memory we had at the end of validation
      tgn.memory.restore_memory(val_memory_backup)

    new_nodes_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))

    # Save temporary results to disk
    pickle.dump({
      "val_aps": val_aps,
      "new_nodes_val_aps": new_nodes_val_aps,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info(
      'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    logger.info(
      'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

    # Early stopping and checkpointing
    stop_training = early_stopper.early_stop_check(val_ap)
    current_checkpoint = get_checkpoint_path(epoch)
    torch.save(tgn.state_dict(), current_checkpoint)
    if early_stopper.best_epoch == epoch:
      best_checkpoint_path = current_checkpoint

    if stop_training:
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      best_checkpoint_path = best_model_path
      tgn.load_state_dict(torch.load(best_model_path, map_location=device))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      tgn.eval()
      break

  # Training has finished, we have loaded the best model, and we want to backup its current
  # memory (which has seen validation edges) so that it can also be used when testing on unseen
  # nodes
  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()

  ### Test
  tgn.embedding_module.neighbor_finder = full_ngh_finder
  test_ap, test_auc = eval_edge_prediction(model=tgn,
                                                              negative_edge_sampler=test_rand_sampler,
                                                              data=test_data,
                                                              n_neighbors=NUM_NEIGHBORS,
                                                              progress=True, desc="Test")

  if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)

  # Test on unseen nodes
  nn_test_ap, nn_test_auc = eval_edge_prediction(model=tgn,
                                                                          negative_edge_sampler=nn_test_rand_sampler,
                                                                          data=new_node_test_data,
                                                                          n_neighbors=NUM_NEIGHBORS,
                                                                          progress=True, desc="NewNode Test")

  logger.info(
    'Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
  logger.info(
    'Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))
  # Save results for this run
  pickle.dump({
    "val_aps": val_aps,
    "new_nodes_val_aps": new_nodes_val_aps,
    "test_ap": test_ap,
    "new_node_test_ap": nn_test_ap,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

  logger.info('Saving TGN model')
  if USE_MEMORY:
    # Restore memory at the end of validation (save a model which is ready for testing)
    tgn.memory.restore_memory(val_memory_backup)
  torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
  if best_checkpoint_path is None:
    best_checkpoint_path = get_checkpoint_path(early_stopper.best_epoch)
  metadata = {
    "saved_at": time.time(),
    "model_path": str(Path(MODEL_SAVE_PATH).resolve()),
    "best_checkpoint": str(Path(best_checkpoint_path).resolve()) if best_checkpoint_path else None,
    "dataset": DATA,
    "data_dir": str(DATA_DIR.resolve()),
    "torch_version": torch.__version__,
    "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    "model_hyperparams": {
      "node_dim": NODE_DIM,
      "time_dim": TIME_DIM,
      "use_memory": USE_MEMORY,
      "embedding_module": args.embedding_module,
      "message_function": args.message_function,
      "memory_updater": args.memory_updater,
      "aggregator": args.aggregator,
      "message_dim": MESSAGE_DIM,
      "memory_dim": MEMORY_DIM,
      "n_layers": NUM_LAYER,
      "n_heads": NUM_HEADS,
      "dropout": DROP_OUT,
      "n_neighbors": NUM_NEIGHBORS,
      "learning_rate": LEARNING_RATE,
    },
  }
  with open(MODEL_METADATA_PATH, "w") as meta_fp:
    json.dump(metadata, meta_fp, indent=2)
  logger.info('TGN model saved')
  logger.info(f"Model metadata written to {MODEL_METADATA_PATH}")
