#!/usr/bin/env python3
"""Extract Keras/QKeras model metadata for the CI comparison dashboard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract Keras/QKeras model metadata for dashboards.")
    parser.add_argument("--model", required=True, help="Path to .h5/.keras model")
    parser.add_argument("--output", required=True, help="Path to output JSON metadata")
    parser.add_argument("--inputs", default=None, help="Optional NumPy inputs for quick model evaluation")
    parser.add_argument("--labels", default=None, help="Optional true labels for quick model evaluation")
    return parser.parse_args()


def normalize_shape(shape) -> str | list:
    """Convert a Keras shape object into JSON-friendly values.

    Args:
        shape: Shape value returned by Keras, or ``None``.

    Returns:
        A string or nested list that can be serialized as JSON.
    """
    if shape is None:
        return "N/A"
    if isinstance(shape, (list, tuple)):
        if shape and isinstance(shape[0], (list, tuple)):
            return [normalize_shape(item) for item in shape]
        return [None if value is None else int(value) for value in shape]
    return str(shape)


def quantizer_config(config: dict) -> dict[str, str]:
    """Extract compact quantizer labels from a Keras layer config.

    Args:
        config: Layer configuration dictionary returned by Keras.

    Returns:
        Mapping from quantized attribute name to a short display label.
    """
    quantizers = {}
    for key in ("activation", "kernel_quantizer", "bias_quantizer", "depthwise_quantizer"):
        value = config.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            class_name = value.get("class_name") or value.get("module") or "quantizer"
            inner = value.get("config") or {}
            bits = inner.get("bits")
            integer = inner.get("integer")
            if bits is not None and integer is not None:
                quantizers[key] = f"{class_name}({bits},{integer})"
            else:
                quantizers[key] = str(class_name)
        elif isinstance(value, str) and value:
            quantizers[key] = value
    return quantizers


def compute_accuracy(model, inputs_path: Path, labels_path: Path) -> dict[str, float | int | str]:
    """Compute quick classification accuracy when test vectors are available.

    Args:
        model: Loaded Keras/QKeras model.
        inputs_path: Path to NumPy input samples.
        labels_path: Path to NumPy class labels or one-hot labels.

    Returns:
        Accuracy metrics, or an empty dictionary when inputs are missing or empty.
    """
    import numpy as np

    if not inputs_path.exists() or not labels_path.exists():
        return {}

    x_test = np.load(inputs_path)
    y_true = np.load(labels_path)
    y_pred = model.predict(x_test, verbose=0)

    if y_true.ndim > 1 and y_true.shape[-1] > 1:
        true_classes = np.argmax(y_true, axis=-1)
    else:
        true_classes = y_true.reshape(-1)
    pred_classes = np.argmax(y_pred, axis=-1)
    sample_count = min(len(true_classes), len(pred_classes))
    if sample_count == 0:
        return {}

    accuracy = float(np.mean(pred_classes[:sample_count] == true_classes[:sample_count]))
    return {
        "model_test_accuracy": accuracy,
        "model_test_samples": int(sample_count),
        "model_test_accuracy_unit": "fraction",
    }


def main() -> int:
    """Load a model, collect metadata, and write a JSON file."""
    args = parse_args()
    model_path = Path(args.model)
    output_path = Path(args.output)

    from qkeras.utils import load_qmodel

    model = load_qmodel(model_path)
    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)

    layers = []
    for layer in model.layers:
        config = layer.get_config()
        layers.append(
            {
                "name": layer.name,
                "class_name": layer.__class__.__name__,
                "output_shape": normalize_shape(getattr(layer, "output_shape", None)),
                "params": int(layer.count_params()),
                "quantizers": quantizer_config(config),
            }
        )

    metadata = {
        "model_name": model.name,
        "model_file": model_path.name,
        "model_params": int(model.count_params()),
        "model_input_shape": normalize_shape(model.input_shape),
        "model_output_shape": normalize_shape(model.output_shape),
        "model_layer_count": len(model.layers),
        "model_layers": layers,
        "model_summary": "\n".join(summary_lines),
    }
    if args.inputs and args.labels:
        metadata.update(compute_accuracy(model, Path(args.inputs), Path(args.labels)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote model metadata: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
