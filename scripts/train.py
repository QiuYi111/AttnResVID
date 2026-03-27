"""Training script for AttnResVID using Hydra configuration."""

import os
import datetime
import pathlib

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset

from source.dataset_collection import DeepVIDv2Dataset
from source.network_collection import DeepVIDv2
from source.worker_collection import DeepVIDv2Worker
from source.utils import JsonSaver, log_model_info
from source.attnres.config import AttnResConfig


def cfg_to_namespace(cfg: DictConfig):
    """Convert a flat or nested DictConfig to a simple argparse-style namespace."""
    import argparse
    flat = {}
    for group in ("dataset", "network", "attnres", "worker", "pruning"):
        if group in cfg:
            flat.update(OmegaConf.to_container(cfg[group], resolve=True))
    # Also include top-level keys (e.g. from _self_)
    for k, v in cfg.items():
        if k not in ("dataset", "network", "attnres", "worker", "pruning", "hydra"):
            flat[k] = v
    return argparse.Namespace(**flat)


def process_cfg(cfg: DictConfig) -> DictConfig:
    """Resolve dynamic defaults that depend on other config values."""
    # Determine if data source is a folder
    noisy_data_paths = list(cfg.dataset.noisy_data_paths)
    if os.path.isdir(noisy_data_paths[0]):
        is_folder = True
        root = noisy_data_paths[0]
        filenames = sorted(os.listdir(root))
        noisy_data_paths = [os.path.join(root, f) for f in filenames]
        OmegaConf.update(cfg, "dataset.noisy_data_paths", noisy_data_paths)
    else:
        is_folder = False

    OmegaConf.update(cfg, "dataset.is_folder", is_folder, merge=True)

    # Auto learning rate
    if cfg.worker.learning_rate is None:
        lr = 5e-6 if is_folder else 1e-4
        OmegaConf.update(cfg, "worker.learning_rate", lr)

    # Auto batch_per_step
    if cfg.worker.batch_per_step is None:
        bps = 360 if is_folder else 12
        OmegaConf.update(cfg, "worker.batch_per_step", bps)

    # Derived: number of input channels
    n = cfg.dataset.input_pre_post_frame
    OmegaConf.update(cfg, "dataset.in_channels", n * 2 + 1 + 4, merge=True)

    # Auto model_string
    if cfg.worker.model_string is None:
        run_uid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        OmegaConf.update(cfg, "worker.model_string", run_uid)

    return cfg


def build_attnres_config(cfg: DictConfig) -> AttnResConfig:
    a = cfg.attnres
    if not a.enabled:
        return AttnResConfig(enabled=False)
    return AttnResConfig(
        enabled=True,
        mode=a.mode,
        history_len=a.history_len,
        temperature=a.temperature,
        gate_init=a.gate_init,
        score_fn=a.score_fn,
        gate_type=a.gate_type,
        detach_history=a.detach_history,
        fusion_mode=a.fusion_mode,
        share_proj=a.share_proj,
        bottleneck_start_idx=a.bottleneck_start_idx,
        decoder_blocks=1 if a.decoder_enabled else 0,
    )


def run_worker(cfg: DictConfig):
    args = cfg_to_namespace(cfg)

    # Promote dataset/network/worker/wandb keys to top-level on args
    # (DeepVIDv2 and worker expect flat namespace)
    for group in ("dataset", "network", "worker", "pruning"):
        group_cfg = getattr(cfg, group, None)
        if group_cfg is not None:
            for k, v in OmegaConf.to_container(group_cfg, resolve=True).items():
                setattr(args, k, v)

    # Flatten wandb config with "wandb_" prefix so worker can pick it up
    wandb_cfg = cfg.get("wandb", None)
    if wandb_cfg is not None:
        for k, v in OmegaConf.to_container(wandb_cfg, resolve=True).items():
            setattr(args, f"wandb_{k}", v)

    attnres_config = build_attnres_config(cfg)

    # Create datasets
    print("Creating dataset...")
    import argparse
    dataset_list = []
    for path in cfg.dataset.noisy_data_paths:
        args_local = argparse.Namespace(**vars(args))
        args_local.noisy_data_path = path
        dataset_list.append(DeepVIDv2Dataset(args_local))
    dataset = ConcatDataset(dataset_list)

    # Create network
    print("Creating network...")
    network = DeepVIDv2(args, attnres_config=attnres_config)

    # Log model info
    print("Logging model info...")
    output_dir = os.path.join(
        pathlib.Path(__file__).parent.absolute(), "..", "results", cfg.worker.model_string
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "config"), exist_ok=True)
    log_model_info(network, output_dir, input_size=(1, args.in_channels, 64, 64))

    # Save resolved config
    config_path = os.path.join(output_dir, "config", "config_train.yaml")
    OmegaConf.save(cfg, config_path)
    # Also save JSON for backward compatibility with inference script
    json_path = os.path.join(output_dir, "config", "config_train.json")
    JsonSaver(OmegaConf.to_container(cfg, resolve=True)).save_json(json_path)

    # Apply Wanda pruning before training if requested
    pruning_cfg = cfg.get("pruning", None)
    pruner = None
    if pruning_cfg and pruning_cfg.enabled and pruning_cfg.get("prune_after_epoch", 0) == 0:
        from source.pruning.wanda import WandaPruner
        print(f"Applying Wanda pruning (sparsity={pruning_cfg.sparsity})...")
        pruner = WandaPruner(network, cfg=pruning_cfg)
        pruner.prune(dataset, args)

    # Create and run worker
    print("Creating worker...")
    args.output_dir = output_dir
    trainer = DeepVIDv2Worker(dataset, network, args)
    print("Running worker...")
    trainer.run()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = process_cfg(cfg)
    run_worker(cfg)


if __name__ == "__main__":
    main()
