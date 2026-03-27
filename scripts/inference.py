"""Inference script for AttnResVID using Hydra configuration."""

import os
import argparse
import pathlib

import hydra
from omegaconf import DictConfig, OmegaConf

from source.dataset_collection import DeepVIDv2Dataset
from source.network_collection import DeepVIDv2
from source.worker_collection import DeepVIDv2Worker
from source.utils import JsonLoader, JsonSaver
from source.attnres.config import AttnResConfig


def load_training_cfg(model_string: str) -> DictConfig:
    """Load training config saved during training (YAML or JSON fallback)."""
    results_root = os.path.join(
        pathlib.Path(__file__).parent.absolute(), "..", "results", model_string
    )
    yaml_path = os.path.join(results_root, "config", "config_train.yaml")
    json_path = os.path.join(results_root, "config", "config_train.json")

    if os.path.exists(yaml_path):
        return OmegaConf.load(yaml_path)
    elif os.path.exists(json_path):
        loader = JsonLoader(json_path)
        loader.load_json()
        return OmegaConf.create(loader.json_data)
    else:
        raise FileNotFoundError(
            f"No training config found for model '{model_string}'. "
            f"Expected {yaml_path} or {json_path}."
        )


def build_attnres_config(cfg: DictConfig) -> AttnResConfig:
    a = cfg.get("attnres", None)
    if a is None or not a.get("enabled", False):
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
        share_proj=a.get("share_proj", False),
        bottleneck_start_idx=a.get("bottleneck_start_idx", 2),
        decoder_blocks=1 if a.get("decoder_enabled", False) else 0,
    )


def run_worker(cfg: DictConfig):
    model_string = cfg.model_string
    results_root = os.path.join(
        pathlib.Path(__file__).parent.absolute(), "..", "results", model_string
    )

    # Load training config and merge inference overrides on top
    train_cfg = load_training_cfg(model_string)
    # Inference-specific fields override training config
    merged = OmegaConf.merge(train_cfg, cfg)

    # Resolve data paths
    noisy_data_paths = list(merged.get("noisy_data_paths", merged.dataset.noisy_data_paths
                                       if "dataset" in merged else []))
    if "noisy_data_paths" in cfg:
        noisy_data_paths = list(cfg.noisy_data_paths)

    if os.path.isdir(noisy_data_paths[0]):
        root = noisy_data_paths[0]
        noisy_data_paths = [os.path.join(root, f) for f in sorted(os.listdir(root))]

    # Default model path
    model_path = cfg.get("model_path", None) or os.path.join(
        results_root, "checkpoint", "checkpoint_final.ckpt"
    )

    attnres_config = build_attnres_config(merged)

    # Build flat args namespace for dataset/network/worker
    flat = {}
    for group in ("dataset", "network", "worker"):
        if group in merged:
            flat.update(OmegaConf.to_container(merged[group], resolve=True))
    flat.update(OmegaConf.to_container(merged, resolve=True))
    flat["output_dir"] = results_root
    flat["model_path"] = model_path
    flat["model_string"] = model_string
    args_base = argparse.Namespace(**flat)

    # Apply Wanda pruning at inference time if explicitly requested
    pruning_cfg = merged.get("pruning", None)
    if pruning_cfg and pruning_cfg.get("enabled", False):
        from source.pruning.wanda import WandaPruner
        # Need a temporary dataset to collect calibration activations
        _tmp_args = argparse.Namespace(**vars(args_base))
        _tmp_args.noisy_data_path = noisy_data_paths[0]
        _tmp_dataset = DeepVIDv2Dataset(_tmp_args)
        _tmp_network = DeepVIDv2(_tmp_args, attnres_config=attnres_config)
        print(f"[Inference] Applying Wanda pruning (sparsity={pruning_cfg.sparsity})...")
        pruner = WandaPruner(_tmp_network, cfg=pruning_cfg)
        pruner.prune(_tmp_dataset, args_base)
        del _tmp_dataset, pruner

    for noisy_data_path in noisy_data_paths:
        args_local = argparse.Namespace(**vars(args_base))
        args_local.noisy_data_path = noisy_data_path

        # Set output filename
        output_file = cfg.get("output_file", None)
        if output_file is None:
            root_p, _ = os.path.splitext(noisy_data_path)
            input_filename_no_ext = os.path.basename(root_p)
            output_file = os.path.join(
                results_root,
                f"{model_string}_result_{input_filename_no_ext}.tiff",
            )
        args_local.output_file = output_file
        args_local.output_dtype = cfg.get("output_dtype", "float32")

        print("Creating dataset...")
        dataset = DeepVIDv2Dataset(args_local)

        print("Creating network...")
        network = DeepVIDv2(args_local, attnres_config=attnres_config)

        print("Creating worker...")
        trainer = DeepVIDv2Worker(dataset, network, args_local)

        print("Running worker...")
        trainer.run()

    # Save inference config
    os.makedirs(os.path.join(results_root, "config"), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(results_root, "config", "config_inference.yaml"))


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Inference requires model_string
    if not cfg.get("model_string", None) and not cfg.worker.get("model_string", None):
        raise ValueError("model_string must be provided for inference. "
                         "Use: python scripts/inference.py model_string=<name>")
    model_string = cfg.get("model_string", None) or cfg.worker.model_string
    OmegaConf.update(cfg, "model_string", model_string, merge=True)
    run_worker(cfg)


if __name__ == "__main__":
    main()
