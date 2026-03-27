"""
https://github.com/AllenInstitute/deepinterpolation/blob/master/deepinterpolation/generic.py
"""

import json
from dataclasses import asdict, is_dataclass

import torch
import torch.nn as nn


class JsonLoader:
    """
    JsonLoader is used to load the data from all structured json files associated with the DeepInterpolation package.
    """

    def __init__(self, path):
        self.path = path

        self.load_json()

    def load_json(self):
        """
        This function load the json file from the path recorded in the class instance.
        Parameters:
        None
        Returns:
        None
        """

        with open(self.path, "r") as f:
            self.json_data = json.load(f)

    def set_default(self, parameter_name, default_value):
        """
        set default forces the initialization of a parameter if it was not present in
        the json file. If the parameter is already present in the json file, nothing
        will be changed.
        Parameters:
        parameter_name (str): name of the paramter to initialize
        default_value (Any): default parameter value
        Returns:
        None
        """

        if not (parameter_name in self.json_data):
            self.json_data[parameter_name] = default_value

    def get_type(self):
        """
        json types define the general category of the object the json file applies to.
        For instance, the json can apply to a data Generator type
        Parameters:
        None

        Returns:
        str: Description of the json type
        """

        return self.json_data["type"]

    def get_name(self):
        """
        Each json type is sub-divided into different names. The name defines the exact construction logic of the object and how the
        parameters json data is used. For instance, a json file can apply to a Generator type using the AudioGenerator name when
        generating data from an audio source. Type and Name fully defines the object logic.
        Parameters:
        None

        Returns:
        str: Description of the json name
        """

        return self.json_data["name"]


class JsonSaver:
    """
    JsonSaver is used to save dict data into individual file.
    """

    def __init__(self, dict_save):
        self.dict = dict_save

    def save_json(self, path):
        """
        This function save the json file into the path provided.
        Parameters:
        str: path: str
        Returns:
        None
        """
        def convert_to_serializable(obj):
            """Convert non-serializable objects to serializable format."""
            if is_dataclass(obj):
                return asdict(obj)
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)

        serializable_dict = {}
        for key, value in self.dict.items():
            try:
                # Test if value is JSON serializable
                json.dumps(value)
                serializable_dict[key] = value
            except (TypeError, ValueError):
                # Convert if not serializable
                serializable_dict[key] = convert_to_serializable(value)

        if isinstance(self.dict, dict):
            with open(path, "w") as f:
                json.dump(serializable_dict, f, indent=4)
        elif isinstance(self.dict, list):
            with open(path, "w") as f:
                for line in self.dict:
                    json.dump(line, f, indent=4)
                    f.write("\n")


##################################################
# Model Profiling Utilities
##################################################


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def count_parameters_by_module(model: nn.Module) -> dict:
    """Count parameters grouped by module name.

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping module names to parameter counts
    """
    params_by_module = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            module_name = name.rsplit(".", 1)[0] if "." in name else "root"
            params_by_module[module_name] = params_by_module.get(module_name, 0) + param.numel()

    return params_by_module


def estimate_flops(model: nn.Module, input_size: tuple = (1, 11, 64, 64)) -> int:
    """Estimate FLOPs for a model.

    This is a rough estimation based on convolution and linear layer operations.
    For accurate FLOPs counting, consider using torchprofile or fvcore.

    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)

    Returns:
        Estimated FLOPs
    """
    def count_conv_flops(module, x, y):
        # Conv2d: output_elements * kernel_elements * in_channels / out_groups
        batch_size = x[0].shape[0]
        output_elements = y.shape.numel() // batch_size

        if isinstance(module, nn.Conv2d):
            kernel_elements = module.kernel_size[0] * module.kernel_size[1]
            in_channels = module.in_channels
            out_channels = module.out_channels
            groups = module.groups

            flops = output_elements * in_channels * kernel_elements // groups
            return flops * batch_size
        return 0

    def count_linear_flops(module, x, y):
        if isinstance(module, nn.Linear):
            return y.shape.numel() * x[0].shape.numel()
        return 0

    # Register hooks
    hooks = []
    flops = 0
    flops_by_layer = {}

    def hook_fn(name):
        def fn(module, input, output):
            nonlocal flops
            layer_flops = 0

            if isinstance(module, nn.Conv2d):
                batch_size = input[0].shape[0]
                output_elements = output.shape.numel() // batch_size
                kernel_elements = module.kernel_size[0] * module.kernel_size[1]
                in_channels = module.in_channels
                groups = module.groups

                layer_flops = output_elements * in_channels * kernel_elements // groups
                layer_flops *= batch_size

            elif isinstance(module, nn.Linear):
                layer_flops = output.numel() * input[0].shape.numel()

            flops += layer_flops
            if layer_flops > 0:
                flops_by_layer[name] = layer_flops

        return fn

    # Add hooks to all relevant layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)

    # Run a forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        _ = model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return flops


def get_model_info(model: nn.Module, input_size: tuple = (1, 11, 64, 64)) -> dict:
    """Get comprehensive model information.

    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)

    Returns:
        Dictionary containing model information
    """
    info = {
        "total_params": count_parameters(model),
        "trainable_params": count_parameters(model, trainable_only=True),
        "non_trainable_params": count_parameters(model, trainable_only=False) - count_parameters(model, trainable_only=True),
        "params_by_module": count_parameters_by_module(model),
    }

    # Estimate FLOPs
    try:
        info["estimated_flops"] = estimate_flops(model, input_size)
    except Exception as e:
        info["estimated_flops"] = f"Error: {str(e)}"

    # Check for AttnRes info
    if hasattr(model, "get_attnres_info"):
        attnres_info = model.get_attnres_info()
        if attnres_info is not None:
            info["attnres"] = attnres_info

    if hasattr(model, "get_gate_values"):
        gate_values = model.get_gate_values()
        if gate_values is not None:
            info["gate_values"] = gate_values

    return info


def log_model_info(model: nn.Module, output_dir: str, input_size: tuple = (1, 11, 64, 64)):
    """Log model information to a JSON file.

    Args:
        model: PyTorch model
        output_dir: Directory to save the info file
        input_size: Input tensor size (B, C, H, W)
    """
    import os
    from pathlib import Path

    info = get_model_info(model, input_size)

    # Convert non-serializable items
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    info_serializable = make_serializable(info)

    # Save to file
    output_path = Path(output_dir) / "model_info.json"
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(info_serializable, f, indent=4)

    print(f"Model info saved to {output_path}")

    return info


##################################################
# Performance Metrics Utilities
##################################################


def get_gpu_memory_usage() -> dict:
    """Get current GPU memory usage.

    Returns:
        Dictionary with memory statistics in MB
    """
    try:
        import torch.cuda as cuda
        if cuda.is_available():
            allocated = cuda.memory_allocated() / 1024**2  # Convert to MB
            reserved = cuda.memory_reserved() / 1024**2
            max_allocated = cuda.max_memory_allocated() / 1024**2
            max_reserved = cuda.max_memory_reserved() / 1024**2
            return {
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "max_allocated_mb": max_allocated,
                "max_reserved_mb": max_reserved,
            }
        else:
            return {"error": "CUDA not available"}
    except Exception as e:
        return {"error": str(e)}


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    try:
        import torch.cuda as cuda
        if cuda.is_available():
            cuda.reset_peak_memory_stats()
            cuda.reset_accumulated_memory_stats()
    except Exception:
        pass


class Timer:
    """Simple timer for measuring elapsed time."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0

    def start(self):
        """Start the timer."""
        import time
        self.start_time = time.time()

    def stop(self):
        """Stop the timer and record elapsed time."""
        import time
        if self.start_time is not None:
            self.end_time = time.time()
            self.elapsed += self.end_time - self.start_time
            self.start_time = None

    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is not None:
            # Timer is running, include current elapsed time
            import time
            return self.elapsed + (time.time() - self.start_time)
        return self.elapsed


def measure_inference_time(
    model: nn.Module,
    input_size: tuple = (1, 11, 64, 64),
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Measure inference time for a model.

    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        num_runs: Number of runs to average over
        warmup_runs: Number of warmup runs
        device: Device to run on

    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    model.to(device)

    dummy_input = torch.randn(input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # Synchronize
    if device == "cuda":
        torch.cuda.synchronize()

    # Time inference
    import time
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    total_time = end_time - start_time

    avg_time = total_time / num_runs

    return {
        "num_runs": num_runs,
        "total_time_ms": total_time * 1000,
        "avg_time_ms": avg_time * 1000,
        "fps": 1.0 / avg_time if avg_time > 0 else 0,
        "device": device,
    }


def log_training_metrics(
    step: int,
    loss: float,
    batch_size: int,
    timer: Timer = None,
    output_dir: str = None,
) -> dict:
    """Log training metrics.

    Args:
        step: Current training step
        loss: Current loss value
        batch_size: Batch size
        timer: Optional Timer object
        output_dir: Optional directory to save metrics

    Returns:
        Dictionary with metrics
    """
    metrics = {
        "step": step,
        "loss": loss,
        "batch_size": batch_size,
    }

    # Add timing
    if timer is not None:
        metrics["elapsed_time_s"] = timer.get_elapsed()

    # Add GPU memory
    gpu_mem = get_gpu_memory_usage()
    if "error" not in gpu_mem:
        metrics["gpu_allocated_mb"] = gpu_mem["allocated_mb"]
        metrics["gpu_max_allocated_mb"] = gpu_mem["max_allocated_mb"]

    # Save to file if output directory provided
    if output_dir is not None:
        import os
        from pathlib import Path

        metrics_path = Path(output_dir) / "training_metrics.jsonl"
        os.makedirs(output_dir, exist_ok=True)

        with open(metrics_path, "a") as f:
            import json
            f.write(json.dumps(metrics) + "\n")

    return metrics
