# Retinal structure and lesion evidence

This is a standalone research component; it is not imported by or required for
the main DR classification pipeline. It provides:

- A heuristic green-channel local-contrast vessel-candidate mask.
- An optional yellow-white bright-region candidate that may correspond to the
  optic disc.
- Overlays for those algorithmic candidates and separately supplied retinal
  annotation masks.
- Binary-mask metrics (Dice, IoU, sensitivity, specificity, and pixel
  accuracy) for comparisons against supplied ground-truth annotations.

```python
from src.lesion_retinal_analysis import (
    evaluate_binary_mask,
    localize_optic_disc,
    save_evidence_visualization,
    segment_vessels,
)

vessels = segment_vessels("fundus.jpg")
disc = localize_optic_disc("fundus.jpg")
save_evidence_visualization(
    "fundus.jpg",
    "reports/retinal-evidence.png",
    vessel_mask=vessels,
    optic_disc=disc,
    annotated_regions={"exudate-like": exudate_annotation_mask},
)
metrics = evaluate_binary_mask(vessels, vessel_ground_truth_mask)
```

The image-processing functions are heuristic visualization aids and are not
validated clinical segmenters or detectors. In particular, the optic-disc
output is only a bright-region candidate, and this package does **not**
automatically detect microaneurysms, exudates, or hemorrhages. Lesion-like
overlays are accepted only as caller-supplied annotation masks; use an
appropriately documented and licensed dataset and retain its annotation
provenance. Metrics describe only the supplied prediction/annotation pair and
must not be reported as a validated performance result without a suitable
evaluation protocol and representative data. No accuracy figures are
prepopulated or inferred.

These image-analysis outputs are separate from classifier output and Grad-CAM.
Grad-CAM indicates regions that influence a selected model score; algorithmic
candidate masks indicate image-processing responses; supplied masks represent
annotations with their own provenance. None is proof of a lesion, a diagnosis,
or a substitute for assessment by a qualified eye-care professional. Optional
analysis should be invoked separately and must not gate or interrupt classifier
inference.
