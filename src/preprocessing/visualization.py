"""Display dataset samples with their stored labels."""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
from PIL import Image

from src.preprocessing.dataset import DatasetRecord


def show_samples(
    records: list[DatasetRecord], count: int = 8, seed: int = 42
) -> plt.Figure:
    """Show a reproducible random selection, titled with raw and mapped labels."""
    if count < 1:
        raise ValueError("count must be at least 1.")
    available = [record for record in records if record.image_path is not None]
    if len(available) != len(records):
        raise ValueError("Resolve missing images before displaying dataset samples.")
    if not available:
        raise ValueError("No records with resolved image paths are available.")
    selected = random.Random(seed).sample(available, k=min(count, len(available)))
    columns = min(4, len(selected))
    rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for axis, record in zip(axes_list, selected):
        with Image.open(record.image_path) as image:
            axis.imshow(image.convert("RGB"))
        title = f"{record.sample_id}\nlabel: {record.label}"
        if record.target_label != record.label:
            title += f"\nmapped: {record.target_label}"
        axis.set_title(title)
        axis.axis("off")
    for axis in axes_list[len(selected) :]:
        axis.axis("off")

    figure.tight_layout()
    plt.show()
    return figure
