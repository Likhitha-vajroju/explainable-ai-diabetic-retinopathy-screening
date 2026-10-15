"""Gradient-based visualizations for the experimental DR classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.dr_classification.classifier import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ClassificationConfig,
)
from src.preprocessing.image_pipeline import preprocess_image


GRADCAM_WARNING = (
    "Grad-CAM visualizes model attention and is not proof of a clinical lesion."
)


@dataclass(frozen=True)
class GradCAMResult:
    """JSON-serializable explanation metadata and the image-sized attention map."""

    explanation: dict[str, object]
    heatmap: np.ndarray


def _target_layer(model: nn.Module, architecture: str) -> nn.Module:
    if architecture in ("resnet18", "resnet50"):
        layer4 = getattr(model, "layer4", None)
        if layer4 is not None:
            return layer4[-1].conv2
    elif architecture == "efficientnet_b0":
        features = getattr(model, "features", None)
        if features is not None:
            return features[-1]
    raise ValueError(
        f"Could not select a Grad-CAM convolutional layer for {architecture!r}."
    )


def generate_gradcam(
    image_path: str | Path,
    model: nn.Module,
    config: ClassificationConfig,
    device: torch.device,
    output_path: str | Path,
    target_class_index: int | None = None,
) -> GradCAMResult:
    """Generate an image-sized Grad-CAM map and a three-panel visualization.

    By default, the Grad-CAM target is the model's predicted class. An explicit
    class index may be supplied to inspect that class's gradient-based attention.
    """
    if target_class_index is not None and not 0 <= target_class_index < config.num_classes:
        raise ValueError(
            f"target_class_index must be between 0 and {config.num_classes - 1}."
        )

    source_path = Path(image_path)
    visualization_path = Path(output_path)
    if source_path.resolve() == visualization_path.resolve():
        raise ValueError("Visualization output path must not overwrite the input image.")

    processed = preprocess_image(source_path, config.preprocessing)
    original = np.asarray(processed.original_image, dtype=np.uint8)
    pixels = np.ascontiguousarray(processed.normalized_image.transpose(2, 0, 1))
    pixels = np.ascontiguousarray((pixels - IMAGENET_MEAN) / IMAGENET_STD)
    input_tensor = torch.from_numpy(pixels).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    feature_maps: list[torch.Tensor] = []
    target_layer = _target_layer(model, config.architecture)

    def capture_activation(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        del module, inputs
        feature_maps.append(output)

    hook = target_layer.register_forward_hook(capture_activation)
    was_training = model.training
    model.eval()
    try:
        logits = model(input_tensor)
        if logits.ndim != 2 or logits.shape != (1, config.num_classes):
            raise ValueError(
                "Grad-CAM expects model logits with shape "
                f"(1, {config.num_classes}); got {tuple(logits.shape)}."
            )
        if not feature_maps or not isinstance(feature_maps[0], torch.Tensor):
            raise ValueError("The selected Grad-CAM layer did not produce a tensor.")

        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())
        target_index = (
            predicted_index if target_class_index is None else target_class_index
        )
        gradients = torch.autograd.grad(
            logits[0, target_index], feature_maps[0], retain_graph=False
        )[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        activation_map = torch.relu((weights * feature_maps[0]).sum(dim=1, keepdim=True))
        activation_map = F.interpolate(
            activation_map,
            size=processed.normalized_image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        activation_map = activation_map - activation_map.min()
        maximum = activation_map.max()
        if maximum > 0:
            activation_map = activation_map / maximum
        heatmap = F.interpolate(
            activation_map[None, None],
            size=(processed.original_image.height, processed.original_image.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        heatmap_array = heatmap.detach().cpu().numpy().astype(np.float32)
    finally:
        hook.remove()
        model.train(was_training)

    class_names = config.class_names
    predicted_class = {
        "index": predicted_index,
        "label": class_names[predicted_index],
    }
    target_class = {"index": target_index, "label": class_names[target_index]}
    confidence = float(probabilities[predicted_index].detach().cpu())

    heatmap_rgb = plt.get_cmap("magma")(heatmap_array)[..., :3]
    overlay = np.clip(
        0.58 * (original.astype(np.float32) / 255.0) + 0.42 * heatmap_rgb,
        0.0,
        1.0,
    )

    visualization_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(original)
    axes[0].set_title("Original fundus image")
    axes[1].imshow(heatmap_array, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("Grad-CAM model attention")
    axes[2].imshow(overlay)
    axes[2].set_title("Attention overlay")
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(
        f"Predicted: {predicted_class['label']}  |  "
        f"Class probability: {confidence:.1%} (uncalibrated)",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        GRADCAM_WARNING,
        ha="center",
        fontsize=9,
        color="#8a1c1c",
    )
    try:
        figure.savefig(visualization_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)

    return GradCAMResult(
        explanation={
            "predicted_class": predicted_class,
            "confidence": confidence,
            "target_class": target_class,
            "visualization_path": str(visualization_path),
            "warning": GRADCAM_WARNING,
        },
        heatmap=heatmap_array,
    )
