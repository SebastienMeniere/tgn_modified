from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import os
from typing import Optional

import numpy as np
import torch

from utils.utils import AdvancedNegativeSampler, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics
from model.tgn import TGN


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train_self_supervised.py"

# Resolve a stable default data directory.
# In containers we default to /workspace/data/processed to align with other images.
# Locally, fall back to path relative to this file to preserve existing behavior.
_env_default_data = os.getenv("TGN_DATA_DIR")
if _env_default_data:
    DEFAULT_DATA_DIR = Path(_env_default_data).expanduser().resolve()
else:
    DEFAULT_DATA_DIR = (ROOT / "../data/processed").resolve()

DEFAULT_SAVED_MODELS = ROOT / "saved_models"


def _prompt(prompt: str, default: Optional[str] = None) -> str:
    if default is not None:
        resp = input(f"{prompt} [{default}]: ").strip()
        return resp or default
    return input(f"{prompt}: ").strip()


def _prompt_bool(prompt: str, default: bool = False) -> bool:
    default_str = "y" if default else "n"
    resp = input(f"{prompt} (y/n) [{default_str}]: ").strip().lower()
    if not resp:
        return default
    return resp in {"y", "yes"}


def _resolve_data_dir(user_input: str) -> Path:
    candidate = Path(user_input).expanduser()
    if candidate.is_absolute():
        return candidate
    return (DEFAULT_DATA_DIR / candidate).expanduser()


def train_flow() -> None:
    print("\n=== Train New TGN Model ===")
    dataset = _prompt("Dataset name", "retail")
    data_hint = f"(absolute or relative to {DEFAULT_DATA_DIR})"
    data_input = _prompt(f"Data directory {data_hint}", dataset)
    data_dir = _resolve_data_dir(data_input)
    if not data_dir.exists():
        print(f"Data directory '{data_dir}' does not exist. Aborting training.")
        return

    prefix = _prompt("Model prefix", dataset)
    n_epoch = int(_prompt("Number of epochs", "5"))
    batch_size = int(_prompt("Batch size", "200"))
    use_memory = _prompt_bool("Use node memory?", True)

    node_dim = _prompt("Node embedding dimension (--node_dim)", "4")
    time_dim = _prompt("Time embedding dimension (--time_dim)", "4")
    message_dim = _prompt("Message dimension (--message_dim)", "4")
    memory_dim = _prompt("Memory dimension (--memory_dim)", "4")
    n_layer = _prompt("Number of layers (--n_layer)", "1")

    embedding_module = _prompt("Embedding module [graph_attention/graph_sum/identity/time]", "graph_attention")
    message_function = _prompt("Message function [mlp/identity]", "identity")
    memory_updater = _prompt("Memory updater [gru/rnn]", "gru")
    aggregator = _prompt("Aggregator type", "last")

    use_weighted_bce = _prompt_bool("Use weighted BCE?", False)
    neg_loss_weight = _prompt("Negative loss weight (--neg_loss_weight)", "1.0")
    num_neg = _prompt("Negatives per positive (--num_neg)", "1")

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "-d", dataset,
        "--data_dir", str(data_dir),
        "--prefix", prefix,
        "--n_epoch", str(n_epoch),
        "--bs", str(batch_size),
    ]

    if use_memory:
        cmd.append("--use_memory")

    if node_dim:
        cmd.extend(["--node_dim", node_dim])
    if time_dim:
        cmd.extend(["--time_dim", time_dim])
    if message_dim:
        cmd.extend(["--message_dim", message_dim])
    if memory_dim:
        cmd.extend(["--memory_dim", memory_dim])
    if n_layer:
        cmd.extend(["--n_layer", n_layer])

    if embedding_module:
        cmd.extend(["--embedding_module", embedding_module])
    if message_function:
        cmd.extend(["--message_function", message_function])
    if memory_updater:
        cmd.extend(["--memory_updater", memory_updater])
    if aggregator:
        cmd.extend(["--aggregator", aggregator])

    if use_weighted_bce:
        cmd.append("--use_weighted_bce")
    if neg_loss_weight:
        cmd.extend(["--neg_loss_weight", neg_loss_weight])
    if num_neg:
        cmd.extend(["--num_neg", num_neg])

    print("\nRunning:", " ".join(cmd))
    try:
        # Run from the TGN source directory so relative paths in the
        # training script (saved_models/, log/, results/) align with this repo.
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    except subprocess.CalledProcessError as exc:
        print(f"Training failed with return code {exc.returncode}.")


def _load_metadata(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_dir():
        raise ValueError("Metadata path points to a directory. Provide the .json file or prefix.")
    if path.suffix != ".json":
        # treat as prefix
        path = DEFAULT_SAVED_MODELS / f"{path_str}.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata file '{path}' not found.")
    return path


def _instantiate_model(metadata: dict, device: torch.device):
    config = metadata.get("config", {})
    mh = metadata.get("model_hyperparams", {})
    data_dir = Path(metadata["data_dir"]).expanduser()
    dataset = metadata["dataset"]

    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data(
        dataset,
        different_new_nodes_between_val_and_test=config.get("different_new_nodes", False),
        randomize_features=config.get("randomize_features", False),
        data_dir=data_dir,
    )

    train_ngh_finder = get_neighbor_finder(train_data, config.get("uniform", False))
    full_ngh_finder = get_neighbor_finder(full_data, config.get("uniform", False))

    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_time_statistics(
        full_data.sources, full_data.destinations, full_data.timestamps
    )

    tgn = TGN(
        neighbor_finder=train_ngh_finder,
        node_features=node_features,
        edge_features=edge_features,
        device=device,
        n_layers=mh.get("n_layers", 1),
        n_heads=mh.get("n_heads", 1),
        dropout=mh.get("dropout", 0.1),
        use_memory=mh.get("use_memory", False),
        message_dimension=mh.get("message_dim", 32),
        memory_dimension=mh.get("memory_dim", 128),
        memory_update_at_start=not config.get("memory_update_at_end", False),
        embedding_module_type=mh.get("embedding_module", "graph_attention"),
        message_function=mh.get("message_function", "identity"),
        aggregator_type=mh.get("aggregator", "last"),
        memory_updater_type=mh.get("memory_updater", "gru"),
        n_neighbors=mh.get("n_neighbors"),
        use_destination_embedding_in_message=config.get("use_destination_embedding_in_message", False),
        use_source_embedding_in_message=config.get("use_source_embedding_in_message", False),
        dyrep=config.get("dyrep", False),
        node_dimension=mh.get("node_dim"),
        time_dimension=mh.get("time_dim"),
        mean_time_shift_src=mean_time_shift_src,
        std_time_shift_src=std_time_shift_src,
        mean_time_shift_dst=mean_time_shift_dst,
        std_time_shift_dst=std_time_shift_dst,
    )

    tgn = tgn.to(device)
    tgn.embedding_module.neighbor_finder = full_ngh_finder

    return tgn, {
        "node_features": node_features,
        "edge_features": edge_features,
        "full_data": full_data,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "new_node_val_data": new_node_val_data,
        "new_node_test_data": new_node_test_data,
        "train_ngh_finder": train_ngh_finder,
        "full_ngh_finder": full_ngh_finder,
        "config": config,
        "hyperparams": mh,
    }


def inference_flow() -> None:
    print("\n=== Load Model & Demo Link Prediction ===")
    meta_input = _prompt("Metadata prefix (.json) or path", "test")
    try:
        metadata_path = _load_metadata(meta_input)
    except Exception as exc:
        print(f"Could not load metadata: {exc}")
        return

    metadata = json.loads(metadata_path.read_text())
    model_path = Path(metadata["model_path"]).expanduser()
    if not model_path.exists():
        print(f"Model state dict not found at {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tgn, context = _instantiate_model(metadata, device)
    except Exception as exc:
        print(f"Failed to rebuild model: {exc}")
        return

    try:
        state_dict = torch.load(model_path, map_location=device)
        tgn.load_state_dict(state_dict)
    except Exception as exc:
        print(f"Failed to load state dict: {exc}")
        return

    tgn.eval()
    if metadata.get("model_hyperparams", {}).get("use_memory", False):
        tgn.memory.__init_memory__()

    test_data = context["test_data"]
    if len(test_data.sources) == 0:
        print("Test dataset is empty; cannot run demo.")
        return

    demo_k = min(5, len(test_data.sources))
    sampler_cls = AdvancedNegativeSampler if metadata.get("training_args", {}).get("negative_sampler") == "advanced" else RandEdgeSampler
    neg_sampler = sampler_cls(context["full_data"].sources,
                              context["full_data"].destinations,
                              seed=42)
    _, neg_samples = neg_sampler.sample(demo_k)

    with torch.no_grad():
        pos_prob, neg_prob = tgn.compute_edge_probabilities(
            test_data.sources[:demo_k],
            test_data.destinations[:demo_k],
            neg_samples,
            test_data.timestamps[:demo_k],
            test_data.edge_idxs[:demo_k],
            context["hyperparams"].get("n_neighbors")
        )

    pos_prob = pos_prob.cpu().numpy().reshape(-1)
    neg_prob = neg_prob.cpu().numpy().reshape(-1)

    print("\nSample link predictions (positive vs. sampled negative):")
    for i in range(demo_k):
        src = test_data.sources[i]
        dst = test_data.destinations[i]
        neg_dst = neg_samples[i]
        print(
            f"Interaction {i+1}: customer {src} -> product {dst} | "
            f"P(pos)={float(pos_prob[i]):.4f}, P(neg to {neg_dst})={float(neg_prob[i]):.4f}"
        )


def main():
    while True:
        print("\n=== TGN Utility Menu ===")
        print("  1) Train new model")
        print("  2) Load existing model & demo link prediction")
        print("  3) Exit")
        choice = _prompt("Choose", "3")

        if choice == "1":
            train_flow()
        elif choice == "2":
            inference_flow()
        elif choice == "3":
            print("Goodbye")
            break
        else:
            print("Unknown choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()
