"""
Wanda pruning for CNN models.

Adapts the Wanda method (Sun et al., 2023, "A Simple and Effective Pruning Approach
for Large Language Models") to 2D convolutional networks.

Original idea: prune weights with lowest score = |W_ij| * ||x_j||_2
where x_j is the j-th input activation column (across the calibration set).

For Conv2d with weight shape (C_out, C_in, kH, kW):
  - Treat each (C_in, kH, kW) slice as the "input dimension"
  - Activation norm: L2 norm of the input feature map across spatial positions
    and calibration samples → shape (C_in,), broadcast over kH*kW
  - Pruning score: |W| * activation_norm (broadcast)
  - Zero out lowest-scoring weights
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from omegaconf import DictConfig


class WandaPruner:
    """Prune Conv2d layers using the Wanda criterion.

    Scores each weight by  |W_ij| * ||activation_j||_2  where the activation
    norm is estimated from a calibration set.  Weights with the lowest scores
    are set to zero (unstructured) or entire output channels are removed
    (structured).

    Args:
        model: The nn.Module to prune.
        cfg: Pruning DictConfig (from conf/pruning/wanda.yaml).
    """

    def __init__(self, model: nn.Module, cfg: DictConfig):
        self.model = model
        self.cfg = cfg
        self.sparsity: float = float(cfg.sparsity)
        self.scope: str = cfg.get("scope", "local")
        self.structured: bool = bool(cfg.get("structured", False))
        self.calibration_batches: int = int(cfg.get("calibration_batches", 32))

        # Maps layer_name -> accumulated squared activation sum (C_in,)
        self._act_sq_sum: Dict[str, torch.Tensor] = {}
        self._act_count: Dict[str, int] = {}
        self._hooks: List = []
        self._named_layers: Dict[str, nn.Conv2d] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(self, dataset, args) -> None:
        """Run calibration then apply pruning masks.

        Args:
            dataset: Dataset to sample calibration batches from.
            args: argparse.Namespace with batch_size, num_workers, in_channels.
        """
        self._collect_layers()
        if not self._named_layers:
            print("[Wanda] No Conv2d layers found to prune.")
            return

        print(f"[Wanda] Collecting activations over {self.calibration_batches} batches...")
        self._register_hooks()
        self._run_calibration(dataset, args)
        self._remove_hooks()

        print(f"[Wanda] Applying {'structured' if self.structured else 'unstructured'} "
              f"pruning with sparsity={self.sparsity} (scope={self.scope})...")
        if self.scope == "global":
            self._prune_global()
        else:
            self._prune_local()

        total, pruned = self._count_zeros()
        print(f"[Wanda] Done. {pruned}/{total} weights zeroed "
              f"({100.0 * pruned / max(total, 1):.1f}% sparsity achieved).")

    def get_masks(self) -> Dict[str, torch.Tensor]:
        """Return binary masks (1=kept, 0=pruned) for all pruned layers."""
        masks = {}
        for name, layer in self._named_layers.items():
            masks[name] = (layer.weight.data != 0).float()
        return masks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_layers(self) -> None:
        """Gather all Conv2d layers that match the prune_layers filter."""
        prune_types = list(self.cfg.get("prune_layers", ["Conv2d"]))
        for name, module in self.model.named_modules():
            if any(t == type(module).__name__ for t in prune_types):
                self._named_layers[name] = module

    def _register_hooks(self) -> None:
        """Register forward hooks to accumulate squared input activation norms."""
        for name, layer in self._named_layers.items():
            # Capture name in closure
            def make_hook(layer_name):
                def hook(module, inp, out):
                    x = inp[0]  # (B, C_in, H, W)
                    # Squared L2 norm over B, H, W → (C_in,)
                    sq = x.detach().pow(2).sum(dim=(0, 2, 3))
                    if layer_name not in self._act_sq_sum:
                        self._act_sq_sum[layer_name] = sq.cpu()
                        self._act_count[layer_name] = 1
                    else:
                        self._act_sq_sum[layer_name] += sq.cpu()
                        self._act_count[layer_name] += 1
                return hook
            h = layer.register_forward_hook(make_hook(name))
            self._hooks.append(h)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _run_calibration(self, dataset, args) -> None:
        """Forward a few batches through the model to collect activations."""
        from torch.utils.data import DataLoader, SubsetRandomSampler
        import random

        device = next(self.model.parameters()).device
        n = min(self.calibration_batches * args.batch_size, len(dataset))
        indices = random.sample(range(len(dataset)), n)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(indices),
            num_workers=getattr(args, "num_workers", 0),
        )

        self.model.eval()
        batches_done = 0
        with torch.no_grad():
            for batch in loader:
                if batches_done >= self.calibration_batches:
                    break
                # Dataset returns (noisy, target, mask) or similar tuples
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)
                self.model(x)
                batches_done += 1

    def _activation_norm(self, name: str, c_in: int) -> torch.Tensor:
        """Compute per-channel activation L2 norm from accumulated stats.

        Returns:
            Tensor of shape (C_in,)
        """
        sq_sum = self._act_sq_sum.get(name, torch.zeros(c_in))
        count = max(self._act_count.get(name, 1), 1)
        # Mean squared activation → sqrt gives RMS norm
        return (sq_sum / count).sqrt()

    def _score_layer(self, name: str, weight: torch.Tensor) -> torch.Tensor:
        """Compute Wanda pruning scores for a Conv2d weight tensor.

        Args:
            name: Layer name (for activation lookup).
            weight: Shape (C_out, C_in, kH, kW).

        Returns:
            Score tensor of same shape.
        """
        c_out, c_in, kH, kW = weight.shape
        act_norm = self._activation_norm(name, c_in).to(weight.device)  # (C_in,)
        # Broadcast: (1, C_in, 1, 1) * (C_out, C_in, kH, kW)
        scores = weight.abs() * act_norm.view(1, c_in, 1, 1)
        return scores

    def _prune_local(self) -> None:
        """Prune each layer independently to the target sparsity."""
        for name, layer in self._named_layers.items():
            W = layer.weight.data
            if self.structured:
                self._prune_structured_layer(name, layer)
            else:
                scores = self._score_layer(name, W)
                k = int(self.sparsity * W.numel())
                if k == 0:
                    continue
                threshold = torch.kthvalue(scores.view(-1), k).values
                mask = (scores > threshold).float()
                layer.weight.data.mul_(mask)

    def _prune_global(self) -> None:
        """Prune across all layers with a single global threshold."""
        all_scores = []
        for name, layer in self._named_layers.items():
            scores = self._score_layer(name, layer.weight.data)
            all_scores.append(scores.view(-1))

        all_scores_cat = torch.cat(all_scores)
        total = all_scores_cat.numel()
        k = int(self.sparsity * total)
        if k == 0:
            return
        threshold = torch.kthvalue(all_scores_cat, k).values

        for name, layer in self._named_layers.items():
            scores = self._score_layer(name, layer.weight.data)
            mask = (scores > threshold).float()
            layer.weight.data.mul_(mask)

    def _prune_structured_layer(self, name: str, layer: nn.Conv2d) -> None:
        """Prune entire output channels (filters) with lowest mean score."""
        W = layer.weight.data  # (C_out, C_in, kH, kW)
        scores = self._score_layer(name, W)
        # Mean score per output channel
        channel_scores = scores.mean(dim=(1, 2, 3))  # (C_out,)
        k = int(self.sparsity * W.shape[0])
        if k == 0:
            return
        threshold = torch.kthvalue(channel_scores, k).values
        keep = (channel_scores > threshold)  # (C_out,)
        mask = keep.float().view(-1, 1, 1, 1)
        layer.weight.data.mul_(mask)
        if layer.bias is not None:
            layer.bias.data.mul_(keep.float())

    def _count_zeros(self):
        total = 0
        pruned = 0
        for layer in self._named_layers.values():
            w = layer.weight.data
            total += w.numel()
            pruned += (w == 0).sum().item()
        return total, pruned
