# Explainable AI

The experimental DR classifier supports gradient-based Grad-CAM visualizations
for its `resnet18`, `resnet50`, and `efficientnet_b0` architectures. It uses the
same configured preprocessing as classifier inference and writes a three-panel
image with the original fundus image, Grad-CAM attention map, and attention
overlay.

```python
import torch

from src.dr_classification.classifier import load_model_checkpoint
from src.explainable_ai import generate_gradcam

model, config, device = load_model_checkpoint(
    "models/checkpoints/dr_resnet18/best_model.pt"
)
result = generate_gradcam(
    "path/to/fundus.jpg",
    model,
    config,
    device,
    "reports/gradcam/fundus-gradcam.png",
)

explanation = result.explanation
print(explanation)
print(result.heatmap.shape)  # (original image height, original image width)
```

The explanation contains the predicted class, its maximum softmax probability
(uncalibrated), the class used as the Grad-CAM target, the visualization path,
and an explicit warning. By default, Grad-CAM targets the predicted class; an
optional `target_class_index` can select another of the five classes. The
heatmap represents model attention for that class, not a confirmed lesion,
clinical finding, or proof of a correct prediction. No lesion-level evidence
is produced by this module; any future validated lesion analysis must be
reported separately from Grad-CAM.
